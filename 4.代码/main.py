# -*- coding: utf-8 -*-
"""
bert-base-chinese 多任务训练示例
数据：你提供的2条护理文书
任务1：BIO序列标注（找疼痛、评分、报告等事件）
任务2：事实性分类（判断是不是真疼）
"""

# ============ 第1步：导入需要的工具包 ============
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
from transformers import BertModel, BertTokenizer  # 加载BERT模型和分词器
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器
import numpy as np  # 数值计算库

# ============ 第2步：定义标签体系 ============
# 事件类型（对应你标注的6种）
EVENT_TYPES = ["Pain", "PainAssessment", "PainRelief", "Analgesia", "Report", "Observation"]

# 生成BIO标签列表：O, B-Pain, I-Pain, B-PainAssessment, I-PainAssessment...
BIO_LABELS = ["O"]  # 第0个标签是"O"（其他/无关）
for t in EVENT_TYPES:
    BIO_LABELS.append(f"B-{t}")  # B开头表示事件开始
    BIO_LABELS.append(f"I-{t}")  # I开头表示事件中间/结尾

# 创建标签到数字的映射（模型只能处理数字）
label2id = {label: idx for idx, label in enumerate(BIO_LABELS)}
id2label = {idx: label for idx, label in enumerate(BIO_LABELS)}

print(f"BIO标签数量：{len(BIO_LABELS)}")  # 应该是13个（1个O + 6种事件×2）
print(f"标签映射：{label2id}")

# ============ 第3步：准备训练数据 ============
# 你提供的两条数据，我整理成代码能读的格式


# ============ 第4步：加载BERT分词器 ============
# bert-base-chinese的分词器，会把中文切成单个字
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# 测试分词效果
test_text = "患者诉腹部胀痛"
test_tokens = tokenizer.tokenize(test_text)
print(f"\n分词测试：'{test_text}' → {test_tokens}")


# 输出应该是：['患', '者', '诉', '腹', '部', '胀', '痛']

# ============ 第5步：定义数据集类 ============
class NursingDataset(Dataset):
    """
    数据集类：把原始数据转换成BERT能吃的格式
    继承自PyTorch的Dataset，必须实现__len__和__getitem__
    """

    def __init__(self, data, tokenizer, max_length=128):
        """
        初始化函数
        data: 原始数据列表
        tokenizer: BERT分词器
        max_length: 最大长度（超过截断，不足补齐）
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """返回数据总量"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取一条数据，并转换成BERT需要的格式
        idx: 数据索引
        """
        # 取出原始数据
        item = self.data[idx]
        text = item["text"]  # 文本
        char_labels = item["bio_labels"]  # 字符级别的BIO标签
        factuality = item["factuality"]  # 事实性标签
        negation = item["negation"]  # 否定极性

        # 第1步：用BERT分词器编码文本
        # return_offsets_mapping=True 会返回每个token对应的字符位置
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,  # 最大长度
            padding="max_length",  # 不够就补齐
            truncation=True,  # 超了就截断
            return_offsets_mapping=True,  # 返回位置映射
            return_tensors="pt"  # 返回PyTorch张量
        )

        # 第2步：把字符级BIO标签转成token级标签
        # BERT的tokenizer可能会把某些词分成子词，需要对齐
        token_labels = []
        for offset in encoding["offset_mapping"][0]:
            start, end = offset[0].item(), offset[1].item()

            if start == end:
                # 特殊token（[CLS], [SEP], [PAD]），标签设为-100（忽略）
                token_labels.append(-100)
            else:
                # 取这个token覆盖的所有字符的标签
                # 如果有多个不同标签，取第一个非O的
                labels_in_span = set(char_labels[start:end])
                # 去掉"O"标签
                non_o_labels = [l for l in labels_in_span if l != "O"]
                if non_o_labels:
                    # 取第一个非O标签
                    token_labels.append(label2id[non_o_labels[0]])
                else:
                    # 全是O，就标O
                    token_labels.append(label2id["O"])

        # 第3步：事实性标签转数字
        fact_map = {"factual": 0, "negated": 1, "plan": 2, "conditional": 3,
                    "purpose": 4, "history": 5, "non_event": 6, "relief": 7, "assessment": 8}
        fact_label = fact_map.get(factuality, 0)  # 找不到默认0

        # 第4步：否定极性转数字
        neg_map = {"肯定": 0, "否定": 1}
        neg_label = neg_map.get(negation, 0)

        # 返回处理好的数据
        return {
            "input_ids": encoding["input_ids"][0],  # 输入的token IDs
            "attention_mask": encoding["attention_mask"][0],  # 注意力掩码（1=有效，0=PAD）
            "token_type_ids": encoding["token_type_ids"][0],  # 句子类型（单句全是0）
            "bio_labels": torch.tensor(token_labels),  # BIO标签
            "fact_label": torch.tensor(fact_label),  # 事实性标签
            "neg_label": torch.tensor(neg_label)  # 否定极性标签
        }


# ============ 第6步：定义多任务模型 ============
class MultiTaskBert(nn.Module):
    """
    多任务BERT模型
    1个主任务：BIO序列标注（找事件）
    2个辅助任务：事实性分类、否定极性分类
    """

    def __init__(self, num_bio_labels, num_fact_labels=9, num_neg_labels=2):
        super().__init__()
        # 加载预训练的bert-base-chinese
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        # BERT输出的隐藏层维度是768
        hidden_size = self.bert.config.hidden_size

        # 任务1：BIO序列标注分类器
        # 输入：768维向量 → 输出：13个类别（BIO标签数）
        self.bio_classifier = nn.Linear(hidden_size, num_bio_labels)

        # 任务2：事实性分类器
        # 输入：768维向量（[CLS]位置的输出） → 输出：9个类别
        self.fact_classifier = nn.Linear(hidden_size, num_fact_labels)

        # 任务3：否定极性分类器
        # 输入：768维向量（[CLS]位置的输出） → 输出：2个类别
        self.neg_classifier = nn.Linear(hidden_size, num_neg_labels)

        # 损失函数
        # ignore_index=-100 表示忽略特殊token的损失
        self.loss_fn_bio = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss_fn_cls = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                bio_labels=None, fact_label=None, neg_label=None):
        """
        前向传播函数
        input_ids: 输入的token IDs
        attention_mask: 注意力掩码
        bio_labels: BIO标签（训练时用）
        fact_label: 事实性标签（训练时用）
        neg_label: 否定极性标签（训练时用）
        """
        # 经过BERT得到输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # sequence_output: 每个token的输出 (batch_size, seq_len, 768)
        sequence_output = outputs.last_hidden_state
        # pooled_output: [CLS]位置的输出 (batch_size, 768)
        pooled_output = outputs.pooler_output

        # 任务1：BIO序列标注
        # 对每个token做分类
        bio_logits = self.bio_classifier(sequence_output)  # (batch, seq_len, num_bio)

        # 任务2：事实性分类
        # 对整个句子做分类（用[CLS]的输出）
        fact_logits = self.fact_classifier(pooled_output)  # (batch, num_fact)

        # 任务3：否定极性分类
        neg_logits = self.neg_classifier(pooled_output)  # (batch, num_neg)

        # 计算损失（只在训练时有标签）
        loss = None
        if bio_labels is not None:
            # BIO损失：reshape成(batch*seq_len, num_bio)计算
            loss_bio = self.loss_fn_bio(
                bio_logits.view(-1, bio_logits.size(-1)),
                bio_labels.view(-1)
            )
            # 事实性损失
            loss_fact = self.loss_fn_cls(fact_logits, fact_label)
            # 否定极性损失
            loss_neg = self.loss_fn_cls(neg_logits, neg_label)

            # 总损失 = BIO损失 × 1.0 + 事实性损失 × 0.5 + 否定损失 × 0.3
            # 权重设置：主任务最重要，事实性次之，否定再次
            loss = loss_bio + 0.5 * loss_fact + 0.3 * loss_neg

        return {
            "loss": loss,  # 总损失
            "bio_logits": bio_logits,  # BIO预测结果
            "fact_logits": fact_logits,  # 事实性预测结果
            "neg_logits": neg_logits  # 否定极性预测结果
        }


# ============ 第7步：创建数据集和数据加载器 ============
print("\n正在创建数据集...")

# 创建数据集对象
dataset = NursingDataset(training_data, tokenizer, max_length=128)

# 创建数据加载器
# batch_size=2：一次处理2条数据（我们总共只有2条）
# shuffle=True：打乱顺序（数据多时有用）
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 检查一条数据的样子
sample = dataset[0]
print(f"\n一条数据示例：")
print(f"input_ids形状：{sample['input_ids'].shape}")
print(f"attention_mask形状：{sample['attention_mask'].shape}")
print(f"bio_labels形状：{sample['bio_labels'].shape}")
print(f"事实性标签：{sample['fact_label']}")
print(f"否定极性标签：{sample['neg_label']}")

# ============ 第8步：初始化模型和优化器 ============
print("\n正在初始化模型...")

# 创建多任务模型
model = MultiTaskBert(num_bio_labels=len(BIO_LABELS))

# 定义优化器（AdamW：带权重衰减的Adam）
# lr=2e-5：学习率，BERT微调常用的小学习率
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# ============ 第9步：训练循环 ============
print("\n开始训练...")
print("-" * 50)

# 训练3个epoch（轮次）
num_epochs = 3

for epoch in range(num_epochs):
    print(f"\n第 {epoch + 1}/{num_epochs} 轮训练")

    # 设置为训练模式（启用Dropout等）
    model.train()

    # 记录本轮的总损失
    total_loss = 0

    # 遍历数据加载器
    for batch_idx, batch in enumerate(dataloader):
        # 清零梯度（PyTorch会累积梯度）
        optimizer.zero_grad()

        # 前向传播：把数据喂给模型
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            bio_labels=batch["bio_labels"],
            fact_label=batch["fact_label"],
            neg_label=batch["neg_label"]
        )

        # 获取损失值
        loss = outputs["loss"]

        # 反向传播：计算梯度
        loss.backward()

        # 更新模型参数
        optimizer.step()

        # 累加损失
        total_loss += loss.item()

        # 打印进度
        print(f"  批次 {batch_idx + 1}: 损失 = {loss.item():.4f}")

    # 打印本轮平均损失
    avg_loss = total_loss / len(dataloader)
    print(f"第 {epoch + 1} 轮平均损失: {avg_loss:.4f}")

print("\n训练完成！")
print("-" * 50)

# ============ 第10步：测试模型预测效果 ============
print("\n测试模型预测效果：")

# 设置为评估模式（关闭Dropout等）
model.eval()

# 测试文本
test_texts = [
    "患者诉腹部胀痛，疼痛评分为4分，报告梁嘉琦医生。",
    "护理组长查房示：关注患者疼痛及睡眠，及时做好疼痛干预措施。",
    "患者无腹痛、腹胀。"
]

for text in test_texts:
    print(f"\n原文：{text}")

    # 编码测试文本
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # 预测（不计算梯度）
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            token_type_ids=encoding["token_type_ids"]
        )

    # 解析BIO预测结果
    bio_preds = outputs["bio_logits"].argmax(dim=-1)[0].numpy()

    # 解析事实性预测
    fact_pred = outputs["fact_logits"].argmax(dim=-1).item()
    fact_map_reverse = {0: "事实(真疼)", 1: "否定(无痛)", 2: "计划/宣教",
                        3: "条件", 4: "目的", 5: "既往史", 6: "非事件",
                        7: "缓解", 8: "评分"}

    # 解析否定极性预测
    neg_pred = outputs["neg_logits"].argmax(dim=-1).item()
    neg_map_reverse = {0: "肯定", 1: "否定"}

    # 获取offset mapping来还原位置
    encoding_with_offset = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True
    )
    offsets = encoding_with_offset["offset_mapping"]

    # 找出预测的事件
    predicted_events = []
    current_event = None

    for i, (pred_id, offset) in enumerate(zip(bio_preds, offsets)):
        # 跳过特殊token
        if offset[0] == offset[1]:
            continue

        label = id2label[pred_id]

        if label.startswith("B-"):
            # 事件开始
            if current_event:
                predicted_events.append(current_event)
            event_type = label[2:]  # 去掉"B-"前缀
            current_event = {
                "type": event_type,
                "start": offset[0],
                "end": offset[1]
            }
        elif label.startswith("I-") and current_event:
            # 事件延续
            current_event["end"] = offset[1]
        else:
            # O或其他，结束当前事件
            if current_event:
                predicted_events.append(current_event)
                current_event = None

    # 别忘了最后一个事件
    if current_event:
        predicted_events.append(current_event)

    # 打印预测结果
    print(f"预测事件：")
    for ev in predicted_events:
        # 用offset还原原文中的词
        word = text[ev["start"]:ev["end"]]
        print(f"  - 类型：{ev['type']}，原文：'{word}'，位置：({ev['start']}, {ev['end']})")

    print(f"事实性判断：{fact_map_reverse[fact_pred]}")
    print(f"否定极性：{neg_map_reverse[neg_pred]}")

# ============ 第11步：保存模型 ============
print("\n保存模型...")
torch.save(model.state_dict(), "nursing_bert_model.pth")
print("模型已保存为 nursing_bert_model.pth")

print("\n✅ 全部完成！你可以用这个模型去预测新的护理文书了。")