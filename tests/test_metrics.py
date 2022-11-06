from training.metrics import AccMean


def test_metric():
    metric = AccMean()

    values = [10, 5]
    items = [2, 3]

    for value, num_item in zip(values, items):
        metric(value, num_item)

    value = metric.compute().item()

    assert value == sum(values) / sum(items)
