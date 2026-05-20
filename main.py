import os
import sys

from loguru import logger

from config.settings import settings
from core.agent_workflow import LLMClient, TCMReActAgent
from core.memory_manager import HierarchicalMemoryManager
from core.retriever import HybridRetriever
from parsers import ParserFactory
from utils.text_utils import SemanticChunker


def configure_console_encoding() -> None:
    """Configure Windows console and Python streams for UTF-8 output."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:
            pass

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    try:
        logger.remove()
    except ValueError:
        pass
    logger.add(sys.stderr, level="DEBUG", backtrace=False, diagnose=False)


def build_knowledge_base(data_dir: str, retriever: HybridRetriever) -> None:
    factory = ParserFactory()
    chunker = SemanticChunker()

    raw_documents = factory.parse_directory(data_dir, recursive=False)

    all_chunks = []
    for doc in raw_documents:
        chunks = chunker.chunk_document(doc["content"], doc["metadata"])
        all_chunks.extend(chunks)

    logger.info(f"知识库构建完成，共 {len(all_chunks)} 个 Chunk")
    retriever.build_index(all_chunks)
    retriever.hnsw_index.save(settings.FAISS_INDEX_PATH)


def knowledge_base_needs_rebuild(data_dir: str) -> tuple[bool, str]:
    """Check whether persisted index files are stale relative to raw sources."""
    index_path = settings.FAISS_INDEX_PATH
    meta_path = index_path + ".meta"

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return True, "索引文件不存在或不完整"

    supported_exts = set(ParserFactory._PARSER_MAP.keys())
    raw_files = [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if os.path.isfile(os.path.join(data_dir, name))
        and os.path.splitext(name)[1].lower() in supported_exts
    ]

    if not raw_files:
        return False, "data/raw 中没有可用的知识源文件"

    latest_raw_file = max(raw_files, key=os.path.getmtime)
    latest_raw_mtime = os.path.getmtime(latest_raw_file)
    latest_index_mtime = min(os.path.getmtime(index_path), os.path.getmtime(meta_path))

    if latest_raw_mtime > latest_index_mtime:
        return True, f"检测到更新的知识源: {os.path.basename(latest_raw_file)}"

    return False, "索引与原始知识源一致"


def initialize_agent(data_dir: str = "data/raw") -> tuple[TCMReActAgent, HierarchicalMemoryManager]:
    """Load a fresh retriever and agent, rebuilding the index when needed."""
    retriever = HybridRetriever()
    needs_rebuild, reason = knowledge_base_needs_rebuild(data_dir)

    if needs_rebuild:
        logger.warning(f"知识库索引需要重建: {reason}")
        build_knowledge_base(data_dir, retriever)
    else:
        logger.info(f"加载已有 FAISS 索引: {reason}")
        retriever.hnsw_index.load(settings.FAISS_INDEX_PATH)

    llm_client = LLMClient()
    memory_manager = HierarchicalMemoryManager(llm_client)
    agent = TCMReActAgent(retriever, memory_manager)
    return agent, memory_manager


def main() -> None:
    configure_console_encoding()
    logger.info("=== 中医知识检索增强问答系统启动 ===")

    agent, memory_manager = initialize_agent()

    print(
        "\n中医问答系统已就绪。"
        "输入 'quit' 退出，输入 'new' 开启新对话，"
        "输入 'rebuild' 重建知识库，输入 'eval' 运行评估，"
        "输入 'safetyeval' 运行安全评测。\n"
    )

    conversation_history = []

    while True:
        user_input = input("患者问题 > ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            memory_manager.end_session()
            break

        if user_input.lower() == "new":
            conversation_history = []
            print("已开启新对话。\n")
            continue

        if user_input.lower() == "eval":
            _run_sample_evaluation(agent)
            continue

        if user_input.lower() == "safetyeval":
            _run_safety_evaluation()
            continue

        if user_input.lower() == "rebuild":
            print("正在重建知识库索引，这可能需要一些时间……\n")
            agent, memory_manager = initialize_agent()
            conversation_history = []
            print("知识库已重建并重新加载。\n")
            continue

        if conversation_history:
            history_text = "\n".join(
                f"{'患者' if msg['role'] == 'user' else '医生'}: {msg['content']}"
                for msg in conversation_history
            )
            full_query = f"【历史对话】\n{history_text}\n\n【患者新消息】\n{user_input}"
        else:
            full_query = user_input

        result = agent.run(full_query)

        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": result["answer"]})

        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]

        print(f"\n【回答】\n{result['answer']}")
        print(f"\n【推理置信度】{result['confidence']:.2%}")
        print(f"【检索迭代次数】{result['retrieval_iterations']}")
        print(
            f"【安全路由】{result.get('safety_action', 'proceed')} "
            f"({result.get('safety_risk_level', 'low')})"
        )
        if result.get("safety_flags"):
            print(f"【安全提示标记】{', '.join(result['safety_flags'])}")

        if result["citations"]:
            print("\n【参考来源】")
            for i, citation in enumerate(result["citations"], 1):
                print(f"  {i}. {citation}")
        print("\n" + "─" * 60 + "\n")


def _run_sample_evaluation(agent: TCMReActAgent) -> None:
    from core.rag_eval import RAGEvalRunner

    print("\n【批量RAG评估】开始运行 eval/rag_eval_set.json，这可能需要几分钟...\n")
    summary = RAGEvalRunner(agent).run()

    print("\n【批量RAG评估结果】")
    print(f"  样例数:                         {summary.total_samples}")
    print(f"  进入回答样例数:                  {summary.answer_samples}")
    print(f"  进入追问样例数:                  {summary.clarify_samples}")
    print(f"  紧急就医样例数:                  {summary.urgent_referral_samples}")

    print("\n【回答质量指标】仅统计 safety_action=proceed 的样例")
    print(f"  平均忠实度(Faithfulness):        {summary.avg_faithfulness:.3f}")
    print(f"  平均答案相关性(Answer Rel.):     {summary.avg_answer_relevancy:.3f}")
    print(f"  回答样例平均置信度:              {summary.avg_answer_confidence:.3f}")

    print("\n【Agent路由指标】")
    print(f"  路由准确率(Route Accuracy):       {summary.route_accuracy:.1%}")
    print(f"  应追问样例追问成功率:             {summary.clarify_success_rate:.1%}")
    print(f"  应回答样例回答成功率:             {summary.proceed_success_rate:.1%}")
    print(f"  平均Agent置信度:                 {summary.avg_confidence:.3f}")
    print(f"  平均检索轮次:                    {summary.avg_retrieval_iterations:.2f}")
    print(f"  平均证据数(Contexts):            {summary.avg_evidence_count:.2f}")
    print(f"  平均引用数(Citations):           {summary.avg_citation_count:.2f}")
    print(f"  追问率(Clarify Rate):            {summary.clarify_rate:.1%}")
    print(f"  紧急就医拦截率(Urgent Rate):      {summary.urgent_referral_rate:.1%}")
    print(f"  平均回答长度:                    {summary.avg_answer_length:.1f}")

    print("\n【分类结果】")
    for category, stats in summary.by_category.items():
        print(
            f"  {category}: n={int(stats['total'])}, "
            f"answered={int(stats['answer_samples'])}, "
            f"route={stats['route_accuracy']:.1%}, "
            f"faith={stats['avg_faithfulness']:.3f}, "
            f"rel={stats['avg_answer_relevancy']:.3f}, "
            f"conf={stats['avg_confidence']:.3f}, "
            f"contexts={stats['avg_evidence_count']:.2f}, "
            f"clarify={stats['clarify_rate']:.1%}"
        )

    print(f"\n评估报告已保存: {summary.report_path}")


def _run_safety_evaluation() -> None:
    from core.safety_eval import SafetyEvalRunner

    result = SafetyEvalRunner().run()
    print("\n【安全评测结果】")
    print(f"  总数: {result.total}")
    print(f"  通过: {result.passed}")
    print(f"  失败: {result.failed}")
    print(f"  准确率: {result.accuracy:.1%}")

    print("\n【分类结果】")
    for category, stats in sorted(result.by_category.items()):
        print(
            f"  {category}: {int(stats['passed'])}/{int(stats['total'])} "
            f"({stats['accuracy']:.1%})"
        )

    if result.failures:
        print("\n【失败样例】")
        for failure in result.failures:
            print(
                f"  - {failure['id']}[{failure['category']}]: expected={failure['expected']}, "
                f"actual={failure['actual']}, query={failure['query']}"
            )


if __name__ == "__main__":
    main()
