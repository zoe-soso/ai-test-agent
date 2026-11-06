import allure
import pytest
from pages.login_page import LoginPage
from utils.config_reader import load_config
from utils.assertions import assert_contains

config = load_config()


@allure.feature("用户登录功能")
@allure.story("正确账号密码登录成功")
@pytest.mark.usefixtures("ensure_test_account")
@pytest.mark.parametrize("email,password,expected", [
    ("123456@gmail.com", "123456", "zoe"),
])
def test_login_success(page_context, email, password, expected):
    login_page = LoginPage(page_context)

    with allure.step("打开站点首页"):
        login_page.open(config["base_url"])

    with allure.step("打开登录页面"):
        login_page.open_login_page()

    with allure.step("使用已注册账号登录"):
        login_page.login(email, password)

    with allure.step("验证页面顶部显示 Logged in as 用户名"):
        assert_contains(login_page.get_login_user(), expected)

    with allure.step("验证登录成功状态"):
        assert login_page.is_login_success(expected)
