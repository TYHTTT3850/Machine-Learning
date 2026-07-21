import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from torch.optim import AdamW
import json


# ==========================================
# 1. 定义 Dataset
# ==========================================
class QADataset(Dataset):
    def __init__(self, questions, paragraphs):
        self.questions = questions
        self.paragraphs = paragraphs

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        q_item = self.questions[idx]
        p_id = q_item['paragraph_id']

        # 因为 paragraphs 是列表，且 p_id 是整数，直接作为索引取值
        paragraph = self.paragraphs[p_id]

        return q_item, paragraph


# ==========================================
# 2. 定义 collate_fn
# ==========================================
def create_collate_fn(tokenizer, max_length=384):
    def collate_fn(batch):
        questions_batch = [item[0]['question_text'] for item in batch]
        paragraphs_batch = [item[1] for item in batch]

        # 1. 批量进行分词与编码
        encodings = tokenizer(
            questions_batch,
            paragraphs_batch,
            max_length=max_length,
            truncation="only_second",  # 仅截断超长的段落
            padding="max_length",
            return_offsets_mapping=True,  # 返回 Token 与字符位置的映射关系
            return_tensors="pt"
        )

        start_positions = []
        end_positions = []

        for i, (q_item, paragraph) in enumerate(batch):
            # 获取真实的字符级别起止索引
            char_start = q_item['answer_start']
            char_end = q_item['answer_end']

            sequence_ids = encodings.sequence_ids(i)
            offset = encodings["offset_mapping"][i]

            # 2. 找到段落 (context) 在 token 序列中的物理起止边界
            context_start = 0
            while sequence_ids[context_start] != 1:
                context_start += 1

            context_end = len(sequence_ids) - 1
            while sequence_ids[context_end] != 1:
                context_end -= 1

            # 3. 判断答案是否因为段落超长被截断而丢失
            # 注意这里的 char_end 判断，因为 Tokenizer 返回的是左闭右开区间
            if offset[context_start][0] > char_start or offset[context_end][1] <= char_end:
                # 答案丢失，将预测目标指向 [CLS] 标记（索引为 0）
                start_positions.append(0)
                end_positions.append(0)
            else:
                # 寻找答案的 start_token 位置
                idx = context_start
                # 只要当前 token 的“结束字符位置” <= 答案真实的“起始字符位置”，继续往后找
                while idx <= context_end and offset[idx][1] <= char_start:
                    idx += 1
                start_positions.append(idx)

                # 寻找答案的 end_token 位置
                idx = context_end
                # 只要当前 token 的“起始字符位置” > 答案真实的“结束字符位置”，继续往前找
                while idx >= context_start and offset[idx][0] > char_end:
                    idx -= 1
                end_positions.append(idx)

        # 5. 移除不需要参与模型前向传播的 offsets_mapping
        encodings.pop("offset_mapping")

        # 6. 将计算好的标签加入到最终的输出字典中
        encodings.update({
            'start_positions': torch.tensor(start_positions, dtype=torch.long),
            'end_positions': torch.tensor(end_positions, dtype=torch.long)
        })

        return encodings

    return collate_fn


# ==========================================
# 3. 主程序：训练、验证与保存
# ==========================================
if __name__ == "__main__":
    # --- A. 准备数据 (模拟训练集和验证集) ---
    train_data = json.load(open('Q&A_train.json', encoding='utf-8'))
    train_data_questions = train_data['questions']
    train_data_paragraphs = train_data['paragraphs']
    validation_data = json.load(open('Q&A_validation.json', encoding='utf-8'))
    validation_data_questions = validation_data['questions']
    validation_data_paragraphs = validation_data['paragraphs']

    my_cache_dir = "./my_models"
    best_model_save_dir = "./best_qa_model"  # 最佳模型保存路径
    os.makedirs(best_model_save_dir, exist_ok=True)

    # --- B. 加载分词器和模型 ---
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', cache_dir=my_cache_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained('bert-base-chinese', cache_dir=my_cache_dir)

    # --- C. 构建 DataLoader ---
    collate_fn = create_collate_fn(tokenizer, max_length=512)

    train_dataset = QADataset(train_data_questions, train_data_paragraphs)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    val_dataset = QADataset(validation_data_questions, validation_data_paragraphs)
    # 验证集不需要 shuffle，batch_size 可以设置得比训练集大一些（因为不需要计算梯度，显存占用小）
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)

    # --- D. 训练环境设置 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)
    epochs = 5

    # 用于记录最佳的验证集 Loss
    best_val_loss = float('inf')

    print(f"使用设备: {device}")
    print("开始训练...")

    # --- E. 训练与验证循环 ---
    for epoch in range(epochs):
        # ========================
        # 1. 训练阶段
        # ========================
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask,
                token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions
            )

            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # ========================
        # 2. 验证阶段
        # ========================
        model.eval()
        total_val_loss = 0

        #验证时不需要计算梯度
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)

                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    token_type_ids=token_type_ids, start_positions=start_positions, end_positions=end_positions
                )

                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # ========================
        # 3. 保存最佳模型
        # ========================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"发现更好的模型 (Val Loss 降至 {best_val_loss:.4f})，正在保存...")

            # 保存模型权重和配置文件
            model.save_pretrained(best_model_save_dir)
            # 同时也保存分词器，确保以后加载时使用的是完全一致的分词规则
            tokenizer.save_pretrained(best_model_save_dir)

    print("训练结束！最佳模型已保存在:", best_model_save_dir)