from BinaryOptionsTools.platforms.pocketoption.stable_api import PocketOption


ssid = r"""42["auth",{"session":"a:4:{s:10:\"session_id\";s:32:\"d4bb9c38855d6210c7ad76fa49938095\";s:10:\"ip_address\";s:10:\"102.67.4.7\";s:10:\"user_agent\";s:111:\"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36\";s:13:\"last_activity\";i:1730100230;}b7de7a6e63635cd0fa5ff8bd3424627a","isDemo":0,"uid":67934497,"platform":2}]"""
account = PocketOption(ssid, True)
account.connect()
msg2 = account.check_connect()
print(msg2)
balance = account.get_balance()
print(balance)
