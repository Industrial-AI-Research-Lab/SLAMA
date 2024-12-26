import pickle

from examples.spark.examples_utils import get_spark_session
from sparklightautoml.transformers.scala_wrappers.target_encoder_transformer import TargetEncoderTransformer


def main():
    spark = get_spark_session()

    # with open("/home/nikolay/Downloads/2ee9e228-1457-46e2-aeaf-d11f2ce7d0f5.pickle_str", "rb") as f:
    with open("/home/nikolay/Downloads/06e63523-cfa4-4bb0-a925-ff1190e7c6b0.pickle_str", "rb") as f:
        data = f.read()

    result = pickle.loads(data)

    tet = TargetEncoderTransformer.create(**result)


if __name__ == "__main__":
    main()
