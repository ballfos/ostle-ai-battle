from ostle.agents.random import RandomAgent
from ostle.app.engine import AsyncEngine
from ostle.app.window import OstleWindow

# 上記のクラスをインポート
# from ostle.app.engine import AsyncEngine
# from ostle.app.renderer import OstleWindow


def main():
    # 1. エージェントの作成
    agent1 = RandomAgent()
    agent2 = RandomAgent()

    # 2. エンジンの初期化
    engine = AsyncEngine(agent1, agent2)

    # 3. ウィンドウの作成と実行
    window = OstleWindow(engine)
    window.run()


if __name__ == "__main__":
    main()
