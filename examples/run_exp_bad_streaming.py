from kubernetes import client, config

exp_name = "test-2"
namespace = "nbutakov"

# Configs can be set in Configuration class directly or using helper utility
config.load_kube_config()


def get_status(exp_name: str):
    v1 = client.CoreV1Api()
    print("Listing pods with their IPs:")
    ret = v1.list_namespaced_pod(namespace=namespace, label_selector=f'runname={exp_name}')
    driver_pod = next((pod for pod in ret.items if pod.metadata.name.endswith('-driver')))
    exec_pod = next((pod for pod in ret.items if pod.metadata.name.endswith('-exec-1')))
    driver_log = v1.read_namespaced_pod_log(name=driver_pod.metadata.name, namespace=namespace)
    executor_log = v1.read_namespaced_pod_log(name=exec_pod.metadata.name, namespace=namespace)
    termination = exec_pod.status.container_statuses[0].state.terminated
    return {
        "exp_name": exp_name,
        "termination": termination.reason if termination else None,
        "driver_log": driver_log,
        "executor_log": executor_log
    }


def run_exp():
    pass

record = get_status(exp_name)
k = 0
# for
#     elt = i.status.container_statuses[0].state.terminated
#     print("%s\t%s\t%s\t%s" % (i.status.pod_ip, i.metadata.namespace, i.metadata.name, elt.reason if elt else None))
