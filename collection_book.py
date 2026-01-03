from playwright.sync_api import sync_playwright
from langchain_classic.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool
import pandas as pd
import os
import time
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()

# 第一步：设置AI Agent（使用环境变量获取 API Key）
# 请确保已在系统中设置环境变量 DEEPSEEK_API_KEY
api_key = os.getenv("DEEPSEEK_API_KEY")

if not api_key:
    raise ValueError("未检测到 API Key。请在 .env 文件中设置 DEEPSEEK_API_KEY，或在系统环境变量中设置它。")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0
)


# 第二步：封装Playwright采集工具（给AI Agent调用）
def crawl_book_data(page_num=1):
    """
    豆瓣图书采集工具，page_num：页码（每页25本）
    """
    # 兼容处理：AI Agent 可能会传入字符串格式的页码，甚至是 "page_num=1" 这种格式
    try:
        if isinstance(page_num, str) and "=" in page_num:
            page_num = page_num.split("=")[-1]
        page_num = int(page_num)
    except (ValueError, TypeError):
        page_num = 1

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        start = (page_num - 1) * 25
        page.goto(f"https://book.douban.com/top250?start={start}")
        page.wait_for_load_state("networkidle")

        book_list = []
        for book_item in page.locator(".item").all():
            title = book_item.locator(".pl2 a").get_attribute("title") or "未知书名"
            author_info = book_item.locator("p.pl").inner_text() or "未知信息"
            rating = book_item.locator(".rating_nums").inner_text() or "0.0"
            book_list.append({
                "书名": title,
                "作者信息": author_info,
                "评分": rating
            })

        browser.close()
        # 返回原始数据字符串，方便AI处理
        return str(book_list)

# 第三步：将采集工具注册给AI Agent
tools = [
    Tool(
        name="DoubanBookCrawler",
        func=crawl_book_data,
        description="用于采集豆瓣读书Top250的图书信息，参数是页码page_num，返回原始图书数据"
    )
]

# 第四步：初始化AI Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # 打印AI Agent的思考过程
    handle_parsing_errors=True  # 自动处理大模型输出格式错误
)

# 第五步：AI Agent执行任务（用户指令自然语言即可）
def ai_book_assistant(user_prompt):
    # 1. AI Agent 自动调用采集工具，获取原始数据
    raw_data = agent.run(user_prompt)
    # 2. AI Agent 智能清洗数据（直接让大模型返回结构化数据）
    clean_prompt = f"请清洗以下图书数据，过滤无效信息，统一格式，返回JSON格式，去重：{raw_data}"
    response = llm.invoke(clean_prompt)
    clean_data = response.content
    # 3. 转换为DataFrame并导出CSV
    import json
    import re
    try:
        # 提取JSON数据（处理大模型返回的格式问题）
        json_match = re.search(r'(\[.*\])', clean_data, re.DOTALL)
        if json_match:
            clean_data_json = json.loads(json_match.group(1))
        else:
            clean_data_json = json.loads(clean_data)
            
        df = pd.DataFrame(clean_data_json)
        df.to_csv("豆瓣图书整理结果.csv", index=False, encoding="utf_8_sig")
        print("数据已导出为 CSV 文件！")

        # 生成书名 txt 文件，文件名包含 8 位 unix 时间戳（精确到分钟）
        timestamp = int(time.time() // 60)
        txt_filename = f"books_{timestamp}.txt"
        with open(txt_filename, "w", encoding="utf-8") as f:
            for item in clean_data_json:
                # 兼容不同的 key 名
                book_name = item.get("书名") or item.get("title") or item.get("book_name")
                if book_name:
                    f.write(f"{book_name}\n")
        print(f"书名已导出为 {txt_filename} 文件！")

        return clean_data_json
    except Exception as e:
        print(f"数据处理失败：{e}")
        return clean_data

# 运行项目（用户自然语言指令）
if __name__ == "__main__":
    user_command = "请采集豆瓣读书Top250的前2页图书信息，筛选评分≥9.0的图书"
    result = ai_book_assistant(user_command)
    print("最终整理结果：")
    for item in result:
        print(item)