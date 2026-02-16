from ostle.agents.inoue_agent.agent import InoueAgent
from ostle.agents.inoue_agent.agent2 import InoueAgentKAI
from ostle.agents.inoue_agent.agent3 import InoueDQNAgent
from ostle.agents.kyawan2 import KyawanAgentV2
from ostle.agents.kyawan3 import KyawanAgentV3
from ostle.app.engine import AsyncEngine
from ostle.app.window import OstleWindow

# 上記のクラスをインポート
# from ostle.app.engine import AsyncEngine
# from ostle.app.renderer import OstleWindow


def main():
    # 1. エージェントの作成
    agent1 = InoueDQNAgent()
    agent2 = KyawanAgentV2()

    # 2. エンジンの初期化
    engine = AsyncEngine(agent1, agent2)

    # 3. ウィンドウの作成と実行
    window = OstleWindow(engine)
    window.run()


if __name__ == "__main__":
    main()
