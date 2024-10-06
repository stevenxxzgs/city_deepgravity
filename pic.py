from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

def save_map_as_image(html_path, img_path, chromedriver_path):
    # 配置Chrome选项
    options = Options()
    options.add_argument('--headless')  # 无头模式
    
    # 设置Service类来使用chromedriver
    service = Service(executable_path=chromedriver_path)
    
    # 启动WebDriver
    driver = webdriver.Chrome(service=service, options=options)
    
    # 设置窗口大小
    driver.set_window_size(1920, 1080)
    
    # 访问地图HTML文件
    driver.get(f"file://{html_path}")
    
    # 截图并保存为图片
    driver.save_screenshot(img_path)
    
    # 关闭WebDriver
    driver.quit()

# 指定chromedriver的路径
chromedriver_path = '/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome'
# 注意：通常chromedriver的路径不是Google Chrome应用的路径，而是一个独立的可执行文件路径。
# 请确保你下载了对应版本的chromedriver，并将其放置在适当的目录下。

# 调用函数保存地图为图片
save_map_as_image("../output_data/output_map.html", "../output_data/output_map.png", chromedriver_path)