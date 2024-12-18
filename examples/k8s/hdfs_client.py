from hdfs import InsecureClient
client = InsecureClient('http://node21.bdcl:9870', user='test')

client.makedirs(hdfs_path="/opt/spark_data", permission=777)

client.upload(
    hdfs_path="/opt/spark_data/sampled_app_train.csv",
    local_path="examples/data/sampled_app_train.csv"
)

client.set_permission(hdfs_path="/opt/spark_data/sampled_app_train.csv", permission=777)

print(client.list("/opt/spark_data"))
