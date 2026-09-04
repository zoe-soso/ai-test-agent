import allure
import pytest
from pages.login_page import LoginPage
from utils.config_reader import load_config
from utils.assertions import assert_contains

config = load_config()

@allure.feature("用户登录功能")
@allure.story("登录功能验证")
@pytest.mark.usefixtures("ensure_test_account")
@pytest.mark.parametrize("email,password,expected_username", [
    ("123456@gmail.com", "123456", "zoe"),
])
def test_login_with_correct_credentials(page_context, email, password, expected_username):
    login_page = LoginPage(page_context)

    with allure.step("打开站点首页"):
        login_page.open(config["base_url"])

    with allure.step("进入登录页面"):
        login_page.open_login_page()

    with allure.step("输入邮箱地址和密码并点击登录按钮"):
        login_page.login(email, password)

    with allure.step("验证登录成功，页面右上角显示用户昵称"):
        assert_contains(login_page.get_login_user(), expected_username)