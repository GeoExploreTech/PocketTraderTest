from BinaryOptionsTools import pocketoption

ssid = (r'42["auth",{"session":"n6ghkt8nk931jj6ffljoj8knj3","isDemo":1,"uid":85249466,"platform":2}]')
ssid2 = (r'42["auth",{"session":"ej3cf6cqp1s3cfols76volj8o7","isDemo":1,"uid":67934497,"platform":2}]')
api = pocketoption(ssid2)

print(api.GetBalance())