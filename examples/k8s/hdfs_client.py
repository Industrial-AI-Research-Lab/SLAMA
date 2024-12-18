from hdfs import InsecureClient
client = InsecureClient('http://node21.bdcl:9870', user='test')
print(client.list("/tmp"))
