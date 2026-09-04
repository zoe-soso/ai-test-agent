import allure
import pytest
from pages.login_page import LoginPage
from utils.config_reader import load_config
from utils.assertions import assert_contains

config = load_config()

@allure.feature("用户登录功能")
@allure.story("登录功能验证")
@pytest.mark.usefixtures("ensure_test_account")
@pytest.mark.parametrize("email,password,expected", [
    ("123456@gmail.com", "123456", "zoe"),
])
def test_login_with_registered_account(page_context, email, password, expected):
    login_page = LoginPage(page_context)

    with allure.step("打开首页"):
        login_page.open(config["base_url"])

    with allure.step("点击登录入口"):
        login_page.open_login_page()

    with allure.step("输入正确的邮箱和密码并点击登录按钮"):
        login_page.login(email, password)

    with allure.step("验证登录成功，页面顶部显示 'Logged in as 用户名'"):
        assert_contains(login_page.get_login_user(), expected)