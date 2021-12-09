import os
import time
from pathlib import Path
from typing import Optional

from neuralogic.core import Template, Backend, Settings, ErrorFunction, Initializer
from utils import Results, to_json, export_fold, ResultList, Crossval


class Evaluator:
    @staticmethod
    def train(model, dataset, epochs):
        outputs = []
        labels = []
        losses = []

        results, _ = model(dataset.samples, train=True, epochs=epochs)

        for result in results:
            outputs.append(result[1])
            labels.append(result[0])
            losses.append(result[2])
        return Results(outputs, labels, loss=losses)

    @staticmethod
    def test(model, dataset):
        outputs = []
        labels = []
        losses = []

        results = model(dataset.samples, train=False)

        for result in results:
            outputs.append(result[1])
            labels.append(result[0])
            losses.append(result[2])
        return Results(outputs, labels, loss=losses)

    @staticmethod
    def learn(model, train_dataset, val_dataset, test_dataset, steps=1000, lr=0.000015):
        best_val_results = Results()
        best_val_results.loss = 1e10
        best_train_results = None
        best_test_results = None

        cumtime = 0

        for epoch in range(steps):
            start = time.time()
            train_results = Evaluator.train(model, train_dataset, 1)
            val_results = Evaluator.test(model, val_dataset)
            test_results = Evaluator.test(model, test_dataset)

            end = time.time()

            if val_results.loss < best_val_results.loss:
                print(f"improving validation loss to {val_results.loss} at epoch {epoch}")
                best_val_results = val_results
                best_test_results = test_results
                best_train_results = train_results
                print(f"storing respective test results with accuracy {best_test_results.accuracy}")

            elapsed = end - start
            cumtime += elapsed

            print(
                "Epoch: {:03d}, Train Loss: {:.7f}, "
                "Train Acc: {:.7f}, Val Acc: {:.7f}, Test Acc: {:.7f}".format(
                    epoch, train_results.loss, train_results.accuracy, val_results.accuracy, test_results.accuracy
                )
                + " elapsed: "
                + str(elapsed)
            )
        return best_train_results, best_val_results, best_test_results, cumtime / steps

    @staticmethod
    def crossvalidate(
        model_string,
        folds,
        outpath: Optional[Path],
        num_node_features: int,
        steps: int = 1000,
        lr: float = 0.000015,
        dim: int = 10,
    ):
        if outpath is None:
            outpath = Path("./out")

        train_results = []
        val_results = []
        test_results = []
        times = []

        for train_fold, val_fold, test_fold in folds:
            template = Evaluator.get_template(model_string)
            settings = Settings(
                initializer=Initializer.GLOROT,
                epochs=steps,
                learning_rate=lr,
                error_function=ErrorFunction.CROSSENTROPY,
            )

            model = template.build(Backend.JAVA, settings)

            built_train_fold = model.build_dataset(train_fold)
            built_val_fold = model.build_dataset(val_fold)
            built_test_fold = model.build_dataset(test_fold)

            best_train_results, best_val_results, best_test_results, elapsed = Evaluator.learn(
                model, built_train_fold, built_val_fold, built_test_fold, steps
            )

            train_results.append(best_train_results)
            val_results.append(best_val_results)
            test_results.append(best_test_results)
            times.append(elapsed)

            train = to_json(ResultList(train_results, times=times))
            export_fold(train, outpath / "train")

            test = to_json(ResultList(test_results))
            export_fold(test, outpath / "test")
        return Crossval(train_results, val_results, test_results, times)

    @staticmethod
    def get_template(string: str):
        dirname = os.path.dirname(os.path.abspath(__file__))

        if string == "gcn":
            return Template(template_file=os.path.join(dirname, "templates", "gcn.txt"))
        if string == "gsage":
            return Template(template_file=os.path.join(dirname, "templates", "gsage.txt"))
        if string == "gin":
            return Template(template_file=os.path.join(dirname, "templates", "gin.txt"))
        raise NotImplementedError
