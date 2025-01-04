import asyncio
import logging
import os.path
from typing import Any, Dict, Optional

import pandas as pd
import tqdm
from kubernetes import client, config
from tqdm.asyncio import tqdm


logger = logging.getLogger(__name__)


BASE_ARTIFACT_FOLDER = "tmp"
NAMESPACE = "nbutakov"


# Configs can be set in Configuration class directly or using helper utility
config.load_kube_config()


def clean_pods():
    v1 = client.CoreV1Api()
    ret = v1.list_namespaced_pod(namespace=NAMESPACE)
    driver_pods = [pod for pod in ret.items if pod.metadata.name.endswith('-driver')]
    for driver_pod in driver_pods:
        logger.info(f"Deleting driver pod: {driver_pod}")
        v1.delete_namespaced_pod(namespace=NAMESPACE, name=driver_pod.metadata.name)


def get_exp_record(exp_name: str) -> Dict[str, Any]:
    v1 = client.CoreV1Api()
    ret = v1.list_namespaced_pod(namespace=NAMESPACE, label_selector=f'runname={exp_name}')
    driver_pods = [pod for pod in ret.items if pod.metadata.name.endswith('-driver')]
    exec_pods = [pod for pod in ret.items if pod.metadata.name.endswith('-exec-1')]

    assert len(driver_pods) == 1
    assert len(exec_pods) == 1

    driver_pod, exec_pod = driver_pods[0], exec_pods[0]

    driver_log = v1.read_namespaced_pod_log(name=driver_pod.metadata.name, namespace=NAMESPACE)
    executor_log = v1.read_namespaced_pod_log(name=exec_pod.metadata.name, namespace=NAMESPACE)
    termination = exec_pod.status.container_statuses[0].state.terminated
    return {
        "exp_name": exp_name,
        "termination": termination.reason if termination else None,
        "driver_log": driver_log,
        "executor_log": executor_log
    }


async def run_exp(sem: asyncio.Semaphore,
                  exec_instances: int,
                  exec_cores: int,
                  dataset_name: str,
                  exp_name: str) -> Optional[str]:
    async with sem:
        try:
            max_cores = exec_instances * exec_cores
            cmd = (f"REPO=node2.bdcl:5000 KUBE_NAMESPACE={NAMESPACE} "
                   f"SLAMA_EXEC_INSTANCES={exec_instances} SLAMA_EXEC_CORES={exec_cores} "
                   f"SLAMA_MAX_CORES={max_cores} SLAMA_RUN_NAME={exp_name} "
                   f"./bin/slamactl.sh submit-job-k8s "
                   f"examples/spark/spark-test-check-lightgbm.py {dataset_name}")

            stdout_path = os.path.join(BASE_ARTIFACT_FOLDER, f"{exp_name}.stdout",)
            stderr_path = os.path.join(BASE_ARTIFACT_FOLDER, f"{exp_name}.stderr",)

            with open(stdout_path, "w") as stdout, open(stderr_path, "w") as stderr:
                proc = await asyncio.create_subprocess_shell(
                    cmd,
                    # stdout=asyncio.subprocess.PIPE,
                    # stderr=asyncio.subprocess.PIPE
                    stdout=stdout,
                    stderr=stderr
                )

            await proc.wait()

            assert proc.returncode == 0
        except:
            logger.warning("Error during exp running", exc_info=True)
            return None

        return exp_name


async def main(max_concurrency: int = 10):
    logger.info(f"Cleaning namespaces {NAMESPACE}")
    clean_pods()

    datasets = [
        "lama_test_dataset",
        "company_bankruptcy_dataset",
        "used_cars_dataset",
        "adv_used_cars_dataset"
    ]

    spark_settings = [
        {"exec_instances": 1, "exec_cores": 1},
        {"exec_instances": 1, "exec_cores": 4},
        {"exec_instances": 2, "exec_cores": 1},
        {"exec_instances": 2, "exec_cores": 2}
    ]

    configs = [
        {
            "dataset_name": dataset_name,
            "exp_name": f"{dataset_name}__{settings['exec_instances']}_{settings['exec_cores']}",
            **settings
        }
        for dataset_name in datasets
        for settings in spark_settings
    ]

    sem = asyncio.Semaphore(10)

    logger.info(f"Running experiments. Num experiments: {len(configs)}. Max concurrency: {max_concurrency}.")

    tasks = [run_exp(sem=sem, **config) for config in configs]

    # exp_names = await tqdm.gather(*tasks)
    exp_names = await tqdm.gather(*tasks)

    logger.info("Finished experiments. Collecting results...")

    records = [get_exp_record(exp_name) for exp_name in exp_names if exp_name]

    logger.info(f"Writing results as a DataFrame in json-format... (records count {len(records)})")
    df = pd.DataFrame(records)
    df.to_json("streaming_runs_exps.json")
    logger.info("All Finished")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(threadName)s %(levelname)s %(filename)s:%(lineno)d %(message)s"
    )
    asyncio.run(main())
