from sklearn import datasets


def get_dataset():
    rcv1 = datasets.fetch_rcv1()
    return rcv1.data, rcv1.target, 103
