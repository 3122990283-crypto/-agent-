#!/usr/bin/env python3
"""
信贷材料预审 Agent · 运行入口

用法:
    python run.py web      # 启动 Web 控制台
    python run.py smoke    # 冒烟测试（验证环境）
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))


def check_api_config():
    import config
    issues = []
    if not config.PADDLE_OCR_TOKEN:
        issues.append("PADDLE_OCR_TOKEN 未配置（飞桨 OCR）")
    if not config.QWEN_API_KEY:
        issues.append("QWEN_API_KEY 未配置（千问 LLM）")
    return issues


def cmd_web():
    print("=" * 60)
    print("  信贷材料预审 Agent · Web 控制台")
    print("=" * 60)
    issues = check_api_config()
    if issues:
        print("\n⚠️  API 配置提示:")
        for issue in issues:
            print(f"   - {issue}")
        print("   请复制 .env.example 为 .env 并填入你的 API Key")
        print("   本地规则引擎可正常使用，OCR/LLM 需配置API后启用\n")

    import uvicorn
    from web_server import app
    print("  上传门户: http://127.0.0.1:8765/portal")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")


def cmd_smoke():
    print("=" * 60)
    print("  信贷材料预审 Agent · 冒烟测试")
    print("=" * 60)

    import config
    BASE = os.path.dirname(os.path.abspath(__file__))
    print(f"\n[1/4] 检查目录...")
    print(f"  项目根目录: {BASE}")
    kb = os.path.join(BASE, "knowledge_base")
    print(f"  知识库: {kb}")

    issues = check_api_config()
    if issues:
        print(f"  ⚠️ API 配置: {len(issues)} 项未配置（可选）")
    else:
        print(f"  ✓ API 配置: OK")

    print(f"\n[2/4] 检查依赖库...")
    deps_ok = True
    for dep in ["cv2", "numpy", "requests", "openai", "sklearn", "networkx", "fastapi", "uvicorn"]:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} 未安装，请 pip install -r requirements.txt")
            deps_ok = False

    print(f"\n[3/4] 检查核心模块...")
    try:
        import layered_rules
        import preaudit_agent
        import web_server
        print(f"  ✓ 核心模块加载成功")
    except Exception as e:
        print(f"  ✗ 模块加载失败: {e}")
        deps_ok = False

    print(f"\n[4/4] 检查知识库...")
    p_dir = os.path.join(kb, "policies")
    r_dir = os.path.join(kb, "regulations")
    pc = len([f for f in os.listdir(p_dir) if f.endswith(".md")]) if os.path.exists(p_dir) else 0
    rc = len([f for f in os.listdir(r_dir) if f.endswith(".md")]) if os.path.exists(r_dir) else 0
    print(f"  ✓ 内部制度: {pc} 份")
    print(f"  ✓ 监管法规: {rc} 份")

    print(f"\n{'='*60}")
    if deps_ok:
        print("  ✓ 冒烟测试通过！运行 python run.py web 启动服务")
    else:
        print("  请先安装依赖: pip install -r requirements.txt")
    print(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="信贷材料预审 Agent")
    subparsers = parser.add_subparsers(dest="command", help="运行模式")
    subparsers.add_parser("web", help="启动 Web 控制台")
    subparsers.add_parser("smoke", help="冒烟测试")
    args = parser.parse_args()

    if args.command == "web":
        cmd_web()
    elif args.command == "smoke":
        cmd_smoke()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
