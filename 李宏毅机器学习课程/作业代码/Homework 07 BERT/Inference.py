import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

def predict_answer(question, context, model, tokenizer, device):
    """单条数据的推理函数"""
    inputs = tokenizer(
        question,
        context,
        max_length=384,
        truncation="only_second",
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    # 把 offset_mapping 单独取出来备用，不送入模型
    # offset_mapping 记录了每个 Token 对应原文的 (字符起点, 字符终点)
    offset_mapping = inputs.pop("offset_mapping")[0]

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()

    # 防御机制：如果没有找到有效答案，或者索引越界
    if start_idx > end_idx or start_idx >= len(offset_mapping) or end_idx >= len(offset_mapping):
        return "", 0, 0

    predict_answer_ids = input_ids[0][start_idx: end_idx + 1]
    predicted_answer = tokenizer.decode(predict_answer_ids, skip_special_tokens=True)

    # 通过 offset_mapping 逆向查找真实的字符索引
    # offset_mapping[start_idx][0] 是答案第一个 Token 在原文中的起始字符索引
    # Tokenizer 的区间是左闭右开的，所以取结尾索引时必须 -1 才能拿到真实的最后一个字符的索引
    char_start = offset_mapping[start_idx][0].item()
    char_end = offset_mapping[end_idx][1].item()-1
    return predicted_answer.replace(" ", ""), char_start, char_end


# ==========================================
# 主程序：读取测试集 -> 批量预测 -> 保存为新 JSON
# ==========================================
if __name__ == "__main__":
    saved_model_dir = "./best_qa_model"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("正在加载模型与分词器...")
    tokenizer = AutoTokenizer.from_pretrained(saved_model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(saved_model_dir)
    model.to(device)
    model.eval()

    input_file = 'Q&A_test.json'
    output_file = 'Q&A_test_predictions.json'

    print(f"正在读取测试数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    questions = test_data['questions']
    paragraphs = test_data['paragraphs']

    total_questions = len(questions)
    print(f"开始批量预测，共 {total_questions} 条数据...")

    #遍历循环
    for i, q_item in enumerate(questions):
        question_text = q_item['question_text']
        p_id = q_item['paragraph_id']
        context = paragraphs[p_id]

        # 接收三个返回值
        pred_ans, char_start, char_end = predict_answer(question_text, context, model, tokenizer, device)

        q_item['answer_text'] = pred_ans
        q_item['answer_start'] = char_start
        q_item['answer_end'] = char_end

        # 每处理 100 条打印一次进度，避免满屏打印
        if (i + 1) % 100 == 0:
            print(f"已处理 {i + 1} / {total_questions} 条...")

    print(f"\n预测完成，正在保存至: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    print("全部任务处理完毕！")