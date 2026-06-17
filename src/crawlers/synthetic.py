"""基于 DeepSeek LLM 的合成医学数据生成器。

六类生成器：
- ConsultationGenerator：问诊记录（对话式）
- TextbookGenerator：医学教材章节
- SymposiumGenerator：学术座谈报告
- CaseGenerator：经典病例报告（CARE 格式）
- PaperGenerator：合成医学研究论文
- DrugManualGenerator：药品/器械手册
"""

from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.crawlers.base import BaseCrawler
from src.llm import create_llm
from src.logger import get_logger

logger = get_logger(__name__)


class ConsultationGenerator(BaseCrawler):
    """生成中文医疗问诊记录（医生-患者对话 + 诊断摘要）。"""

    DEPARTMENTS = [
        ("心内科", ["胸痛", "心悸", "高血压", "气短", "下肢水肿"]),
        ("消化科", ["腹痛", "反酸", "腹泻", "便秘", "便血"]),
        ("呼吸科", ["咳嗽", "咳痰", "发热", "胸闷", "咯血"]),
        ("神经内科", ["头痛", "头晕", "肢体麻木", "失眠", "记忆力减退"]),
        ("内分泌科", ["多饮多尿", "体重下降", "乏力", "手抖", "颈部增粗"]),
        ("骨科", ["腰痛", "膝关节痛", "颈肩痛", "骨折后复诊", "关节肿胀"]),
        ("儿科", ["发热", "咳嗽", "腹泻", "皮疹", "食欲不振"]),
        ("妇产科", ["月经不调", "下腹痛", "异常出血", "孕检咨询", "更年期症状"]),
        ("皮肤科", ["皮疹", "瘙痒", "脱发", "色斑", "皮肤肿物"]),
        ("急诊科", ["急性腹痛", "外伤", "高热", "意识障碍", "中毒"]),
        ("肾内科", ["浮肿", "尿频尿急", "泡沫尿", "腰痛", "高血压"]),
        ("肿瘤科", ["体重减轻", "淋巴结肿大", "疼痛", "乏力", "食欲减退"]),
    ]

    def __init__(self, output_dir: str = "data/md_documents/consultations"):
        super().__init__(output_dir, request_interval=1.5)

    def crawl(self, max_items: int = 50) -> list[Path]:
        logger.info("问诊记录生成开始，目标 %d 份", max_items)
        llm = create_llm(temperature=0.8)
        chain = _consultation_prompt() | llm | StrOutputParser()
        results: list[Path] = []

        for i in range(max_items):
            dept_idx = i % len(self.DEPARTMENTS)
            dept, complaints = self.DEPARTMENTS[dept_idx]
            complaint = complaints[(i // len(self.DEPARTMENTS)) % len(complaints)]
            sub_idx = (i // (len(self.DEPARTMENTS) * len(complaints))) + 1
            variant = i % 3

            filename = f"consultation_{i + 1:03d}_{dept}_{complaint}.md"
            output_path = self.output_dir / filename
            if output_path.exists():
                results.append(output_path)
                continue

            logger.info("生成问诊 [%d/%d]: %s - %s", i + 1, max_items, dept, complaint)
            try:
                self._rate_limit()
                md = chain.invoke(
                    {
                        "department": dept,
                        "complaint": complaint,
                        "sub_index": sub_idx,
                        "variant": variant,
                    }
                )
                md = _clean_output(md)
                if len(md) < 400:
                    logger.warning("问诊内容过短 (%d chars)，重试", len(md))
                    self._rate_limit()
                    md = chain.invoke(
                        {
                            "department": dept,
                            "complaint": complaint,
                            "sub_index": sub_idx + 10,
                            "variant": (variant + 1) % 3,
                        }
                    )
                    md = _clean_output(md)
                output_path.write_text(md, encoding="utf-8")
                results.append(output_path)
            except Exception:
                logger.exception("问诊生成失败: %s", filename)

        logger.info("问诊生成完成: %d/%d", len(results), max_items)
        return results


def _consultation_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """\
你是一位资深全科医生，请生成一份真实、详尽的中文门诊问诊记录。要求：

1. 格式：Markdown，用 # 做标题，## 做小节标题
2. 结构必须包含：
   # 门诊问诊记录 — {department}
   ## 基本信息 (姓名用"患者XX"，年龄合理，性别随机，就诊日期在2024-2025年)
   ## 主诉 (患者原话描述)
   ## 现病史 (起病时间、诱因、症状演变、伴随症状、院外诊治经过)
   ## 既往史 (既往疾病、手术、过敏史、用药史)
   ## 体格检查 (生命体征 T/P/R/BP + 专科查体发现)
   ## 辅助检查 (列出具体项目和结果，含正常值和异常值)
   ## 初步诊断 (1-3条，含ICD-10编码)
   ## 治疗意见 (药物+剂量+用法，或进一步检查建议)
   ## 医生签名 + 日期
3. 对话部分：在 ## 问诊对话 小节中，模拟医生-患者的5-8轮对话
4. 医学术语要准确、专业，检验值要有具体数字和单位
5. 病情要符合{department}的典型疾病谱，主诉与{complaint}相关
6. 每份记录要有不同的细节（这是第{sub_index}个{variant}变体）
7. 总字数 1200-2500 字
8. 只输出 Markdown，不要额外说明。\
""",
            ),
            ("user", "请生成一份{department}门诊问诊记录，主诉与{complaint}相关。"),
        ]
    )


class TextbookGenerator(BaseCrawler):
    """生成临床医学教材章节（结构化 Markdown）。"""

    CHAPTERS = [
        ("01", "心血管系统疾病概论", "cardiology"),
        ("02", "冠状动脉粥样硬化性心脏病", "cardiology"),
        ("03", "心力衰竭的病理生理与治疗", "cardiology"),
        ("04", "消化系统疾病总论", "gastroenterology"),
        ("05", "消化性溃疡与幽门螺杆菌", "gastroenterology"),
        ("06", "炎症性肠病的诊断与治疗", "gastroenterology"),
        ("07", "呼吸系统感染性疾病", "pulmonology"),
        ("08", "慢性阻塞性肺疾病", "pulmonology"),
        ("09", "支气管哮喘的规范化诊疗", "pulmonology"),
        ("10", "糖尿病分型与综合管理", "endocrinology"),
        ("11", "甲状腺疾病的临床诊治", "endocrinology"),
        ("12", "神经系统查体与定位诊断", "neurology"),
        ("13", "脑血管疾病的急诊处理", "neurology"),
        ("14", "常见肿瘤的筛查策略", "oncology"),
        ("15", "肿瘤免疫治疗原理与临床应用", "oncology"),
        ("16", "急性肾损伤与慢性肾脏病", "nephrology"),
        ("17", "抗菌药物的合理应用", "infectious_disease"),
        ("18", "发热待查的临床思维", "infectious_disease"),
        ("19", "外科围手术期管理", "surgery"),
        ("20", "急诊医学：心肺复苏指南", "emergency"),
    ]

    def __init__(self, output_dir: str = "data/md_documents/textbook"):
        super().__init__(output_dir, request_interval=2.0)

    def crawl(self, max_items: int = 20) -> list[Path]:
        logger.info("教材章节生成开始，目标 %d 章", max_items)
        llm = create_llm(temperature=0.6)
        chain = _textbook_prompt() | llm | StrOutputParser()
        results: list[Path] = []

        for chapter_num, title, domain in self.CHAPTERS[:max_items]:
            filename = f"chapter_{chapter_num}_{title}.md"
            output_path = self.output_dir / filename
            if output_path.exists():
                results.append(output_path)
                continue

            logger.info("生成教材 [%s/%d]: %s", chapter_num, max_items, title)
            try:
                self._rate_limit()
                md = chain.invoke({"chapter_num": chapter_num, "title": title, "domain": domain})
                md = _clean_output(md)
                if len(md) < 600:
                    logger.warning("教材内容过短 (%d chars)，重试", len(md))
                    self._rate_limit()
                    md = chain.invoke({"chapter_num": chapter_num, "title": title, "domain": domain})
                    md = _clean_output(md)
                output_path.write_text(md, encoding="utf-8")
                results.append(output_path)
            except Exception:
                logger.exception("教材生成失败: %s", filename)

        logger.info("教材生成完成: %d/%d", len(results), max_items)
        return results


def _textbook_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """\
你是一位资深医学教授，正在编写一本《临床医学精要》教材。请撰写以下章节。

要求：
1. 使用 Markdown 格式，严格遵守标题层级：# 章节标题 → ## 大节 → ### 小节
2. 内容必须专业、准确，包含最新的临床指南和循证医学证据
3. 结构：
   # 第{chapter_num}章 {title}
   ## 概述（定义、流行病学、临床意义）
   ## 病因与发病机制（含分子机制、病理生理）
   ## 临床表现（症状、体征、分型）
   ## 辅助检查（实验室、影像学、特殊检查，含具体数值范围）
   ## 诊断与鉴别诊断（诊断标准、鉴别要点）
   ## 治疗（药物治疗含具体剂量、非药物治疗、手术指征）
   ## 预后与随访
   ## 要点总结（3-5条bullet points）
4. 遇到药物时给出中文通用名和常用剂量范围
5. 遇到数据时给出具体数值、百分比、参考文献引用标记 [1][2] 等
6. 总字数 2000-3500 字
7. 只输出 Markdown，不要额外说明。\
""",
            ),
            ("user", "请撰写第{chapter_num}章：{title}。要专业、详尽，符合临床医学教材标准。"),
        ]
    )


class SymposiumGenerator(BaseCrawler):
    """生成医学学术座谈/研讨会报告。"""

    TOPICS = [
        "肿瘤免疫治疗的新进展与临床转化",
        "人工智能在医学影像诊断中的应用与挑战",
        "精准医学时代的个体化用药策略",
        "抗生素耐药性的全球挑战与应对",
        "心血管疾病一级预防的最新指南解读",
        "数字疗法在慢性病管理中的实践",
        "真实世界研究在药品监管决策中的价值",
        "多学科诊疗(MDT)模式的经验分享",
        "罕见病的诊断困境与政策建议",
        "区块链技术在医疗数据共享中的应用",
        "DRG付费改革对医院管理的影响",
        "医疗大语言模型的伦理与监管框架",
        "新型疫苗技术平台的研发进展",
        "老年共病患者的安全用药管理",
        "远程医疗在基层医疗中的实践与思考",
        "医工交叉：手术机器人技术前沿",
        "急诊预检分诊的人工智能辅助系统",
        "生物标志物在肿瘤早筛中的研究进展",
        "社区获得性肺炎的规范化诊治",
        "心理健康服务的数字化转型",
    ]

    def __init__(self, output_dir: str = "data/md_documents/symposium"):
        super().__init__(output_dir, request_interval=2.0)

    def crawl(self, max_items: int = 20) -> list[Path]:
        logger.info("座谈报告生成开始，目标 %d 份", max_items)
        llm = create_llm(temperature=0.7)
        chain = _symposium_prompt() | llm | StrOutputParser()
        results: list[Path] = []

        for i, topic in enumerate(self.TOPICS[:max_items]):
            filename = f"symposium_{i + 1:02d}_{topic[:30]}.md"
            # 清理文件名
            filename = filename.replace("/", "_").replace("：", "_").replace(" ", "_")
            output_path = self.output_dir / filename
            if output_path.exists():
                results.append(output_path)
                continue

            logger.info("生成座谈 [%d/%d]: %s", i + 1, max_items, topic[:40])
            try:
                self._rate_limit()
                md = chain.invoke({"topic": topic, "index": i + 1})
                md = _clean_output(md)
                if len(md) < 500:
                    self._rate_limit()
                    md = chain.invoke({"topic": topic, "index": i + 1})
                    md = _clean_output(md)
                output_path.write_text(md, encoding="utf-8")
                results.append(output_path)
            except Exception:
                logger.exception("座谈生成失败: %s", filename)

        logger.info("座谈生成完成: %d/%d", len(results), max_items)
        return results


def _symposium_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """\
你是一位著名三甲医院的科室主任，正在整理一份学术座谈会纪要。请生成这份报告。

要求：
1. 使用 Markdown 格式
2. 结构必须包含：
   # 学术座谈会纪要
   ## 会议主题：{topic}
   ## 基本信息（时间: 2024-2025年某月，地点，主办方，主持人，参会专家3-5位含真实可信的姓名和单位）
   ## 会议背景与目的
   ## 主题报告摘要（2-3位专家的报告要点，每位300-500字，含具体数据和研究发现）
   ## 圆桌讨论要点（列出讨论的核心议题和各方观点，含分歧和共识）
   ## 结论与专家共识（3-5条共识意见）
   ## 后续行动计划
   ## 纪要整理人 + 日期
3. 专家发言要有观点、有数据支撑，体现学术深度
4. 讨论要有不同观点的碰撞，真实感强
5. 总字数 1800-3000 字
6. 只输出 Markdown，不要额外说明。\
""",
            ),
            ("user", '请生成关于"{topic}"的学术座谈会纪要。'),
        ]
    )


class CaseGenerator(BaseCrawler):
    """生成经典教学病例报告（CARE 指南格式）。"""

    CASES = [
        ("cardiology", "急性心肌梗死", "急诊PCI术后并发心源性休克"),
        ("cardiology", "感染性心内膜炎", "反复发热伴心脏杂音"),
        ("gastroenterology", "克罗恩病", "回盲部狭窄伴不全肠梗阻"),
        ("gastroenterology", "自身免疫性肝炎", "转氨酶反复升高"),
        ("pulmonology", "肺栓塞", "突发呼吸困难和胸痛"),
        ("pulmonology", "间质性肺病", "进行性呼吸困难伴干咳"),
        ("neurology", "吉兰-巴雷综合征", "进行性四肢无力"),
        ("neurology", "多发性硬化", "反复发作的视力障碍和肢体麻木"),
        ("endocrinology", "糖尿病酮症酸中毒", "以腹痛为首发表现"),
        ("endocrinology", "嗜铬细胞瘤", "阵发性高血压伴头痛心悸"),
        ("nephrology", "狼疮性肾炎", "年轻女性浮肿伴蛋白尿"),
        ("nephrology", "ANCA相关性血管炎", "急进性肾小球肾炎"),
        ("oncology", "肺癌脑转移", "以神经系统症状首诊"),
        ("oncology", "多发性骨髓瘤", "腰背痛伴贫血"),
        ("hematology", "急性早幼粒细胞白血病", "牙龈出血伴发热"),
        ("infectious_disease", "结核性脑膜炎", "头痛发热伴意识障碍"),
        ("rheumatology", "系统性红斑狼疮", "多浆膜腔积液"),
        ("pediatrics", "川崎病", "持续高热伴皮疹"),
        ("dermatology", "Stevens-Johnson综合征", "药物过敏致全身皮肤剥脱"),
        ("emergency", "主动脉夹层", "撕裂样胸背痛"),
    ]

    def __init__(self, output_dir: str = "data/md_documents/cases"):
        super().__init__(output_dir, request_interval=2.0)

    def crawl(self, max_items: int = 20) -> list[Path]:
        logger.info("病例报告生成开始，目标 %d 份", max_items)
        llm = create_llm(temperature=0.7)
        chain = _case_prompt() | llm | StrOutputParser()
        results: list[Path] = []

        for i, (dept, diagnosis, highlight) in enumerate(self.CASES[:max_items]):
            filename = f"case_{i + 1:02d}_{diagnosis}_{highlight[:20]}.md"
            filename = filename.replace("/", "_").replace("：", "_").replace(" ", "_")
            output_path = self.output_dir / filename
            if output_path.exists():
                results.append(output_path)
                continue

            logger.info("生成病例 [%d/%d]: %s - %s", i + 1, max_items, diagnosis, highlight)
            try:
                self._rate_limit()
                md = chain.invoke(
                    {
                        "department": dept,
                        "diagnosis": diagnosis,
                        "highlight": highlight,
                        "index": i + 1,
                    }
                )
                md = _clean_output(md)
                if len(md) < 700:
                    self._rate_limit()
                    md = chain.invoke(
                        {
                            "department": dept,
                            "diagnosis": diagnosis,
                            "highlight": highlight,
                            "index": i + 1,
                        }
                    )
                    md = _clean_output(md)
                output_path.write_text(md, encoding="utf-8")
                results.append(output_path)
            except Exception:
                logger.exception("病例生成失败: %s", filename)

        logger.info("病例生成完成: %d/%d", len(results), max_items)
        return results


def _case_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """\
你是一位三甲医院主任医师，正在为《中华医学杂志》撰写一份经典教学病例报告。请严格按 CARE 指南格式撰写。

要求：
1. 使用 Markdown 格式，标题层级清晰
2. 结构必须包含以下所有小节：
   # 病例报告：{diagnosis} — {highlight}
   ## 摘要（200字以内的病例概要）
   ## 患者基本信息（年龄、性别、职业，非真实信息但合理）
   ## 主诉（患者原话，要生动具体）
   ## 现病史（起病时间线清晰，症状演变详细，院外诊治经过）
   ## 既往史、个人史、家族史
   ## 体格检查（生命体征 + 系统查体阳性发现 + 重要的阴性发现）
   ## 辅助检查
   ### 实验室检查（血常规、生化、免疫、微生物等，含具体数值和正常范围）
   ### 影像学检查（描述具体发现，含CT/MRI/X线/超声等）
   ### 病理检查（如有，描述镜下所见）
   ## 诊疗经过（按时间顺序，含用药方案和剂量、手术过程、病情变化）
   ## 最终诊断与鉴别诊断（列出诊断依据，排除的疾病及理由）
   ## 治疗与转归（出院时状态、随访结果）
   ## 讨论（结合文献分析：为什么本例特殊/典型？诊断难点？治疗选择的循证依据？教学要点？）
   ## 小结（3-5条 take-home messages）
3. 科室：{department}；核心诊断：{diagnosis}
4. 检验检查数据必须具体、合理、范围准确
5. 治疗药物给出通用名和剂量
6. 总字数 2000-3500 字
7. 只输出 Markdown，不要额外说明。\
""",
            ),
            ("user", "请生成一份{department}科的经典教学病例，核心诊断是{diagnosis}。"),
        ]
    )


class PaperGenerator(BaseCrawler):
    """生成合成医学研究论文（PubMed 不可达时的替代方案）。"""

    PAPER_TOPICS = [
        ("深度学习在结直肠癌病理诊断中的应用", "colorectal_cancer", "pathology"),
        ("基于CT影像的肺癌早期筛查模型研究", "lung_cancer", "radiology"),
        ("糖尿病并发症预测的机器学习方法比较", "diabetes", "endocrinology"),
        ("急诊预检分诊系统的AI辅助决策研究", "emergency", "triage"),
        ("抗菌药物耐药性趋势的回顾性分析", "antibiotic_resistance", "infectious_disease"),
        ("乳腺癌新辅助化疗疗效的影像组学预测", "breast_cancer", "oncology"),
        ("老年高血压患者用药依从性影响因素分析", "hypertension", "cardiology"),
        ("脑卒中后认知障碍的早期筛查工具验证", "stroke", "neurology"),
        ("慢性肾脏病患者贫血管理的临床路径优化", "ckd", "nephrology"),
        ("多模态数据融合在胃癌预后预测中的应用", "gastric_cancer", "oncology"),
        ("儿童哮喘规范化管理的社区干预效果评价", "asthma", "pediatrics"),
        ("人工智能辅助皮肤病诊断系统的开发与验证", "dermatology", "dermatology"),
        ("肝硬化门脉高压的无创评估方法研究", "cirrhosis", "hepatology"),
        ("骨科术后感染的病原学特征及危险因素分析", "orthopedic", "surgery"),
        ("产前筛查新技术的卫生经济学评价", "prenatal", "obstetrics"),
        ("精神科药物血药浓度监测的临床意义", "psychiatry", "pharmacology"),
        ("免疫检查点抑制剂相关不良反应的管理策略", "immunotherapy", "oncology"),
        ("可穿戴设备在心房颤动筛查中的价值", "afib", "cardiology"),
        ("中药注射剂安全性监测的真实世界研究", "tcm", "pharmacovigilance"),
        ("医学大语言模型在临床决策支持中的评估", "llm", "medical_informatics"),
    ]

    def __init__(self, output_dir: str = "data/md_documents/papers"):
        super().__init__(output_dir, request_interval=2.0)

    def crawl(self, max_items: int = 20) -> list[Path]:
        logger.info("论文生成开始，目标 %d 篇", max_items)
        llm = create_llm(temperature=0.5)
        chain = _paper_prompt() | llm | StrOutputParser()
        results: list[Path] = []

        for i, (title, domain, field) in enumerate(self.PAPER_TOPICS[:max_items]):
            filename = f"synthetic_paper_{i + 1:02d}_{domain}.md"
            output_path = self.output_dir / filename
            if output_path.exists():
                results.append(output_path)
                continue

            logger.info("生成论文 [%d/%d]: %s", i + 1, max_items, title)
            try:
                self._rate_limit()
                md = chain.invoke({"title": title, "domain": domain, "field": field})
                md = _clean_output(md)
                if len(md) < 800:
                    self._rate_limit()
                    md = chain.invoke({"title": title, "domain": domain, "field": field})
                    md = _clean_output(md)
                output_path.write_text(md, encoding="utf-8")
                results.append(output_path)
            except Exception:
                logger.exception("论文生成失败: %s", filename)

        logger.info("论文生成完成: %d/%d", len(results), max_items)
        return results


def _paper_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """\
你是一位医学研究专家，请撰写一篇结构完整的医学研究论文。

要求：
1. 使用 Markdown 格式，标题层级清晰
2. 结构必须包含：
   # {title}
   ## 摘要（200-300字，含目的、方法、结果、结论）
   ## 关键词（5-8个）
   ## 引言（研究背景、现状、本研究目的）
   ## 材料与方法
   ### 研究对象（纳入/排除标准、样本量、伦理审批）
   ### 研究方法（具体技术路线、观察指标）
   ### 统计学方法（使用的检验方法、P值标准）
   ## 结果（含具体数据和统计分析，P值和置信区间）
   ## 讨论（结果解读、与文献对比、局限性、临床意义）
   ## 结论
   ## 参考文献（5-8条，格式：[1] 作者. 标题. 期刊, 年份.）
3. 研究领域：{field}
4. 数据要具体、合理、有统计意义
5. 总字数 2000-3500 字
6. 只输出 Markdown，不要额外说明。\
""",
            ),
            ("user", "请撰写论文：{title}。研究领域：{field}。"),
        ]
    )


class DrugManualGenerator(BaseCrawler):
    """生成合成药品/器械手册（DailyMed 不可达时的替代方案）。"""

    DRUGS = [
        ("阿司匹林肠溶片", "aspirin", "cardiovascular", "抗血小板聚集"),
        ("盐酸二甲双胍片", "metformin", "endocrinology", "口服降糖药"),
        ("阿托伐他汀钙片", "atorvastatin", "cardiovascular", "调脂药"),
        ("注射用青霉素钠", "penicillin", "antibiotic", "抗生素"),
        ("盐酸氨溴索口服液", "ambroxol", "pulmonology", "祛痰药"),
        ("硝苯地平控释片", "nifedipine", "cardiovascular", "钙通道阻滞剂"),
        ("奥美拉唑肠溶胶囊", "omeprazole", "gastroenterology", "质子泵抑制剂"),
        ("盐酸西替利嗪片", "cetirizine", "allergy", "抗组胺药"),
        ("硫酸氢氯吡格雷片", "clopidogrel", "cardiovascular", "抗血小板药"),
        ("孟鲁司特钠咀嚼片", "montelukast", "pulmonology", "白三烯受体拮抗剂"),
        ("盐酸曲马多缓释片", "tramadol", "pain_management", "镇痛药"),
        ("人胰岛素注射液", "insulin", "endocrinology", "降糖药"),
        ("头孢呋辛酯片", "cefuroxime", "antibiotic", "头孢类抗生素"),
        ("盐酸贝那普利片", "benazepril", "cardiovascular", "ACEI类降压药"),
        ("氟康唑胶囊", "fluconazole", "antifungal", "抗真菌药"),
        ("甲钴胺片", "mecobalamin", "neurology", "神经营养药"),
        ("盐酸氨氯地平片", "amlodipine", "cardiovascular", "钙通道阻滞剂"),
        ("恩替卡韦分散片", "entecavir", "hepatology", "抗病毒药"),
        ("注射用奥沙利铂", "oxaliplatin", "oncology", "抗肿瘤药"),
        ("左甲状腺素钠片", "levothyroxine", "endocrinology", "甲状腺激素"),
        ("雷贝拉唑钠肠溶片", "rabeprazole", "gastroenterology", "质子泵抑制剂"),
        ("利伐沙班片", "rivaroxaban", "cardiovascular", "抗凝药"),
        ("磷酸奥司他韦胶囊", "oseltamivir", "antiviral", "抗病毒药"),
        ("瑞舒伐他汀钙片", "rosuvastatin", "cardiovascular", "调脂药"),
        ("盐酸坦索罗辛缓释胶囊", "tamsulosin", "urology", "α1受体阻滞剂"),
    ]

    def __init__(self, output_dir: str = "data/md_documents/drug_manual"):
        super().__init__(output_dir, request_interval=1.5)

    def crawl(self, max_items: int = 20) -> list[Path]:
        logger.info("药品手册生成开始，目标 %d 份", max_items)
        llm = create_llm(temperature=0.4)
        chain = _drug_prompt() | llm | StrOutputParser()
        results: list[Path] = []

        for i, (name, eng, field, category) in enumerate(self.DRUGS[:max_items]):
            filename = f"drug_{i + 1:02d}_{eng}.md"
            output_path = self.output_dir / filename
            if output_path.exists():
                results.append(output_path)
                continue

            logger.info("生成药品 [%d/%d]: %s", i + 1, max_items, name)
            try:
                self._rate_limit()
                md = chain.invoke(
                    {
                        "name": name,
                        "eng": eng,
                        "field": field,
                        "category": category,
                    }
                )
                md = _clean_output(md)
                if len(md) < 600:
                    self._rate_limit()
                    md = chain.invoke(
                        {
                            "name": name,
                            "eng": eng,
                            "field": field,
                            "category": category,
                        }
                    )
                    md = _clean_output(md)
                output_path.write_text(md, encoding="utf-8")
                results.append(output_path)
            except Exception:
                logger.exception("药品手册生成失败: %s", filename)

        logger.info("药品手册生成完成: %d/%d", len(results), max_items)
        return results


def _drug_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """\
你是一位资深临床药师，正在编写一本《临床药物手册》。请为以下药物撰写详细的药品信息页。

要求：
1. 使用 Markdown 格式
2. 结构必须包含：
   # {name}（{eng}）
   **药物分类**：{category}
   ## 成分与性状（化学名、分子式、外观性状）
   ## 适应症（列出FDA/NMPA批准的适应症，标注证据等级）
   ## 用法用量（成人、儿童、老年人、肝肾功能不全患者的剂量调整）
   ## 药理作用（作用机制、药效学、药代动力学参数：Tmax/Cmax/T1/2/蛋白结合率/代谢途径/排泄）
   ## 禁忌（绝对禁忌、相对禁忌）
   ## 注意事项（特殊人群用药、监测指标、重要警告）
   ## 不良反应（按频率分类：十分常见>10%、常见1-10%、偶见0.1-1%、罕见<0.1%）
   ## 药物相互作用（列出重要的相互作用药物及机制）
   ## 药物过量（症状、处理措施）
   ## 储存条件（温度、避光、有效期）
   ## 剂型规格（列出常见规格和包装）
3. {field} 领域的专业药物信息
4. 数据要具体、准确，剂量和参数要有具体的数值范围
5. 总字数 1800-3000 字
6. 只输出 Markdown，不要额外说明。\
""",
            ),
            ("user", "请为{name}撰写详细的药品信息页。药物类别：{category}。"),
        ]
    )


def _clean_output(text: str) -> str:
    """清理 LLM 输出：去掉开头的 markdown 标记和多余空行。"""
    text = text.strip()
    # 去掉常见的 ```markdown 和 ``` 包装
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
