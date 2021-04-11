import dynet as dy
from neuralogic.nn.dynet import NeuraLogicLayer

from examples.datasets.multiple_examples_no_order_trains import dataset

neuralogic_layer = NeuraLogicLayer(dataset.weights)
trainer = dy.AdamTrainer(neuralogic_layer.model, alpha=0.001)

epochs = 800
printouts = 10
seen_instances = 0
total_loss = 0

for iter in range(epochs):
    if iter > 0 and iter % printouts == 0:
        print(iter, " average loss is:", total_loss / seen_instances)

    seen_instances = 0
    total_loss = 0

    dy.renew_cg(immediate_compute=False, check_validity=False)

    for sample in dataset.samples:
        label = dy.scalarInput(sample.target)
        graph_output = neuralogic_layer(sample)
        loss = dy.squared_distance(graph_output, label)

        total_loss += loss.value()
        loss.backward()
        trainer.update()
        seen_instances += 1

for sample in dataset.samples:
    dy.renew_cg(immediate_compute=False, check_validity=False)

    graph_output = neuralogic_layer.build_sample(sample)
    label = dy.scalarInput(sample.target)

    print(f"label: {label.value()}, output: {graph_output.value()}")
