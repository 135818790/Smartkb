"""
SmartKB 评估脚本 —— 自建 RAG 质量评估（不依赖 Ragas 复杂依赖链）
使用方式：python tests/eval.py
面试要点：每个指标的评估逻辑都是你自己控制的，
          你能说出 Faithfulness/Relevancy/Precision 的具体打分算法。

四项指标：
  Faithfulness       — 回答的每句话是否能在检索到的文档中找到依据
  Answer Relevancy   — 回答是否切中问题
  Context Precision  — 搜回来的文档里，有用的占多少
  Context Recall     — 标准答案里的事实，文档是否覆盖
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from src.core.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# ============================================================
# 1. 配置
# ============================================================
BASE_URL = "http://127.0.0.1:8001"
llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


# ============================================================
# 2. 调用系统 /chat 接口
# ============================================================
def query_system(question: str, timeout: int = 120) -> dict:
    """调用 SmartKB /chat 接口"""
    url = f"{BASE_URL}/chat"
    data = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return {
                "answer": result.get("answer", ""),
                "contexts": [s.get("snippet", s.get("content", str(s))) for s in result.get("sources", [])],
            }
    except Exception as e:
        print(f"  [错误] {e}")
        return {"answer": "", "contexts": []}


# ============================================================
# 3. 评估函数 —— 每个指标独立调 LLM 打分
# ============================================================

def evaluate_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """
    Faithfulness（忠实度）：回答的每句话是否都能在检索文档中找到依据？
    0.0 = 全是编的，1.0 = 每句话都有出处
    """
    if not answer or not contexts:
        return 0.0

    ctx_text = "\n---\n".join(contexts[:3])  # 只看前 3 段

    prompt = f"""你是回答质量评估员。判断以下回答是否忠实于参考文档。

参考文档：
{ctx_text}

用户问题：{question}

系统回答：{answer}

请打分（0-1之间的小数）：
- 1.0 = 回答的每句话都能在文档中找到明确依据
- 0.7 = 大部分有依据，少数推断合理
- 0.3 = 大部分找不依据
- 0.0 = 完全编造或与文档矛盾

只回复一个数字："""

    resp = llm.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10,
    )
    return _parse_score(resp.choices[0].message.content)


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """
    Answer Relevancy（答案相关性）：回答切中问题了吗？有没有答非所问？
    """
    if not answer:
        return 0.0

    prompt = f"""你是回答质量评估员。判断回答是否切中问题。

用户问题：{question}
系统回答：{answer}

打分（0-1）：
- 1.0 = 完全切中问题，直接回答
- 0.7 = 基本相关但有冗余
- 0.3 = 部分跑题
- 0.0 = 完全答非所问

只回复一个数字："""

    resp = llm.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10,
    )
    return _parse_score(resp.choices[0].message.content)


def evaluate_context_precision(question: str, contexts: list[str]) -> float:
    """
    Context Precision（上下文精确度）：搜回来的文档里，有用的占多少？
    """
    if not contexts:
        return 0.0

    ctx_text = "\n---\n".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts))

    prompt = f"""判断以下检索结果中有多少片段对回答问题有用。

问题：{question}

检索到的片段：
{ctx_text}

打分（0-1）：每段有用≈0.33分，全有用=1.0，全无用=0.0
只回复一个数字："""

    resp = llm.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10,
    )
    return _parse_score(resp.choices[0].message.content)


def evaluate_context_recall(question: str, ground_truth: str, contexts: list[str]) -> float:
    """
    Context Recall（上下文召回）：标准答案中的信息，文档覆盖了多少？
    """
    if not contexts or not ground_truth:
        return 0.5  # 无法判断时给中间值

    ctx_text = "\n---\n".join(contexts[:3])

    prompt = f"""判断检索到的文档是否包含回答以下问题所需的关键信息。

问题：{question}
标准答案：{ground_truth}
检索到的文档：
{ctx_text}

打分（0-1）：标准答案的关键信息在文档中能找到多少？
1.0=全部覆盖，0.5=部分覆盖，0.0=完全缺失
只回复一个数字："""

    resp = llm.chat.completions.create(
        model="deepseek-chat", messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=10,
    )
    return _parse_score(resp.choices[0].message.content)


def _parse_score(text: str) -> float:
    """从 LLM 回复中提取数字"""
    import re
    match = re.search(r"(\d+\.?\d*)", str(text).strip())
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return 0.5  # 解析失败给中间值


# ============================================================
# 4. 主评估流程
# ============================================================
def run_evaluation(test_file: str = "tests/test_cases.json"):
    with open(test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"\n{'='*60}")
    print(f"📋 SmartKB RAG 质量评估")
    print(f"{'='*60}")
    print(f"测试用例: {len(test_cases)} 条")
    print(f"评估模型: DeepSeek-V3（低成本）")
    print(f"目标: Faithfulness>0.85, Relevancy>0.85, Precision>0.75")
    print(f"{'='*60}\n")

    results = []
    for i, tc in enumerate(test_cases):
        q = tc["question"]
        gt = tc.get("ground_truth", "")

        print(f"[{i+1:2d}/{len(test_cases)}] {q[:70]}")

        # 调用系统
        r = query_system(q)
        answer = r["answer"]
        contexts = r["contexts"]

        if not answer:
            print(f"      ⚠️  无响应")
            continue

        # 四项评估
        faith = evaluate_faithfulness(q, answer, contexts)
        relev = evaluate_answer_relevancy(q, answer)
        cprec = evaluate_context_precision(q, contexts)
        crecl = evaluate_context_recall(q, gt, contexts)

        results.append({
            "question": q,
            "answer": answer[:200],
            "ground_truth": gt,
            "faithfulness": round(faith, 3),
            "answer_relevancy": round(relev, 3),
            "context_precision": round(cprec, 3),
            "context_recall": round(crecl, 3),
        })

        print(f"      Faith={faith:.2f}  Relev={relev:.2f}  CPrec={cprec:.2f}  CRec={crecl:.2f}")

        # 避免 API 限流
        time.sleep(0.5)

    # ============================================================
    # 5. 汇总报告
    # ============================================================
    if not results:
        print("\n❌ 无有效评估结果")
        return

    avg_f = sum(r["faithfulness"] for r in results) / len(results)
    avg_r = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_p = sum(r["context_precision"] for r in results) / len(results)
    avg_c = sum(r["context_recall"] for r in results) / len(results)
    composite = avg_f * 0.35 + avg_r * 0.25 + avg_p * 0.25 + avg_c * 0.15

    print(f"\n{'='*60}")
    print(f"📊 最终评估报告")
    print(f"{'='*60}")

    def bar(val):
        return "█" * int(val * 20) + "░" * (20 - int(val * 20))

    print(f"  Faithfulness      : {avg_f:.3f}  {bar(avg_f)}")
    print(f"  Answer Relevancy  : {avg_r:.3f}  {bar(avg_r)}")
    print(f"  Context Precision : {avg_p:.3f}  {bar(avg_p)}")
    print(f"  Context Recall    : {avg_c:.3f}  {bar(avg_c)}")
    print(f"  {'─'*50}")
    print(f"  综合得分          : {composite:.3f}")
    grade = "🏆 S" if composite > 0.85 else "👍 A" if composite > 0.75 else "⚠️  B" if composite > 0.65 else "🔧 C"
    print(f"  评级              : {grade}")
    print(f"{'='*60}")

    # 保存报告
    out = {
        "summary": {
            "faithfulness": round(avg_f, 3),
            "answer_relevancy": round(avg_r, 3),
            "context_precision": round(avg_p, 3),
            "context_recall": round(avg_c, 3),
            "composite": round(composite, 3),
            "test_count": len(results),
        },
        "details": results,
    }
    out_path = Path(test_file).parent / "eval_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 报告已保存: {out_path}")


if __name__ == "__main__":
    test_file = sys.argv[1] if len(sys.argv) > 1 else "tests/test_cases.json"
    run_evaluation(test_file)
