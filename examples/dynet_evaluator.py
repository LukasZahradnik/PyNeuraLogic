from examples.datasets.vectorized_xor import template, dataset

from neuralogic.nn import get_evaluator
from neuralogic.core import Backend


neuralogic_evaluator = get_evaluator(template, Backend.DYNET)

printouts = 10

for epoch, (total_loss, seen_instances) in enumerate(neuralogic_evaluator.train(dataset)):
    if epoch % printouts == 0:
        print(
            f"Epoch {epoch}, total loss: {total_loss}, instances: {seen_instances}, average loss {total_loss / seen_instances}"
        )

for label, predicted in neuralogic_evaluator.test(dataset):
    print(f"Label: {label}, predicted: {predicted}")
