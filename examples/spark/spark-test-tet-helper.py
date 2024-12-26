import pickle

from examples_utils import get_spark_session
from sparklightautoml.transformers.scala_wrappers.target_encoder_transformer import TargetEncoderTransformer
from hdfs import InsecureClient


def main():
    spark = get_spark_session()

    client = InsecureClient('http://node21.bdcl:9870', user='test')
    # hdfs_path = "/tmp/tet_dumps/2ee9e228-1457-46e2-aeaf-d11f2ce7d0f5.pickle_str"
    hdfs_path = "/tmp/tet_dumps/06e63523-cfa4-4bb0-a925-ff1190e7c6b0.pickle_str"

    with client.read(hdfs_path) as reader:
        data = reader.read()

    result = pickle.loads(data)

    tet = TargetEncoderTransformer.create(**result)


if __name__ == "__main__":
    main()
