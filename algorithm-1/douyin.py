import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

# 设置目标商品的URL
product_url = "https://www.douyin.com/some-product-link"
# 设置商品上架时间（精确到秒）
launch_time = "2024-10-24 10:00:00"

# 初始化浏览器
chrome_options = Options()
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
chrome_options.add_argument("--window-size=1920,1080")
driver = webdriver.Chrome(executable_path='/path/to/chromedriver', options=chrome_options)

# 打开商品页面
driver.get(product_url)

# 登录抖音商城
def login():
    time.sleep(60)  # 手动登录或扫码登录

# 刷新页面直到指定时间前几秒
def wait_until_launch_time():
    launch_timestamp = time.mktime(time.strptime(launch_time, "%Y-%m-%d %H:%M:%S"))
    while True:
        current_time = time.time()
        time_difference = launch_timestamp - current_time
        if time_difference <= 2:  # 距离商品上架时间还有不到2秒，开始抢购
            print("接近上架时间，开始高频刷新...")
            break
        elif time_difference > 60:
            print(f"离上架时间超过1分钟，当前时间: {current_time}, 上架时间: {launch_timestamp}")
            time.sleep(30)  # 离上架时间超过60秒时，休眠30秒
        elif time_difference > 10:
            print(f"离上架时间超过10秒，当前时间: {current_time}, 上架时间: {launch_timestamp}")
            time.sleep(2)  # 离上架时间超过10秒时，休眠2秒
        else:
            print(f"接近上架时间，当前时间: {current_time}, 上架时间: {launch_timestamp}")
            time.sleep(0.5)  # 最后10秒钟，频繁检查时间

# 刷新页面并检查是否可以购买
def high_freq_refresh():
    while True:
        try:
            buy_button = driver.find_element(By.XPATH, '//*[@id="buy-button"]')
            buy_button.click()  # 点击购买按钮
            print("已点击购买按钮！")
            break
        except:
            print("购买按钮未出现，继续刷新页面...")
            driver.refresh()
            time.sleep(0.1)  # 在最后关键时刻，降低刷新间隔至0.1秒

# 提交订单
def submit_order():
    try:
        submit_button = driver.find_element(By.XPATH, '//*[@id="submit-order-button"]')
        submit_button.click()
        print("订单已提交！")
    except Exception as e:
        print("提交订单失败：", e)

if __name__ == "__main__":
    # 登录抖音商城
    login()

    # 等待商品上架时间到来
    wait_until_launch_time()

    # 高频刷新页面，直到可以购买
    high_freq_refresh()

    # 提交订单
    submit_order()

    # 关闭浏览器
    driver.quit()