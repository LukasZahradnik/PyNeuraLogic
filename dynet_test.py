from dotenv import load_dotenv
import dynet as dy
from neuralogic import data
import neuralogic.dynet as nldy

load_dotenv()


def train():
    model = data.Mutagenesis

    deserializer = nldy.NeuraLogicLayer(model.weights)

    epochs = 400
    trainer = dy.AdamTrainer(deserializer.model, alpha=0.001)
    printouts = 10
    seen_instances = 0
    total_loss = 0

    for iter in range(epochs):
        if iter > 0 and iter % printouts == 0:
            print(iter, " average loss is:", total_loss / seen_instances)

        seen_instances = 0
        total_loss = 0

        dy.renew_cg(immediate_compute=False, check_validity=False)

        losses = []

        for sample in model.samples:
            label = dy.scalarInput(sample.target)
            graph_output = deserializer.build_sample(sample)
            loss = dy.squared_distance(graph_output, label)
            losses.append(loss)

        loss = dy.esum(losses)
        total_loss += loss.value()
        loss.backward()
        trainer.update()
        seen_instances += 1

    for sample in model.samples:
        dy.renew_cg(immediate_compute=False, check_validity=False)

        graph_output = deserializer.build_sample(sample)
        label = dy.scalarInput(sample.target)
        print(f"label: {label.value()}, output: {graph_output.value()}")


train()
