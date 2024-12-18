from hdfs import InsecureClient

dataset_name = "small_used_cars_data.csv"

client = InsecureClient('http://node21.bdcl:9870', user='test')

client.makedirs(hdfs_path="/opt/spark_data", permission=777)

client.upload(
    hdfs_path=f"/opt/spark_data/{dataset_name}",
    local_path=f"examples/data/{dataset_name}"
)

client.set_permission(hdfs_path=f"/opt/spark_data/{dataset_name}", permission=777)

print(client.list("/opt/spark_data"))
