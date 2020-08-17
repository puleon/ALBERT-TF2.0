from absl import app, flags, logging
import os
import csv
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

import classifier_data_lib_mem


FLAGS = flags.FLAGS

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")


flags.DEFINE_string(
    "input_data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")


def main(_):
    processors = {
    "cola": classifier_data_lib_mem.ColaProcessor,
    "sts": classifier_data_lib_mem.StsbProcessor,
    "sst": classifier_data_lib_mem.Sst2Processor,
    "mnli": classifier_data_lib_mem.MnliProcessor,
    "qnli": classifier_data_lib_mem.QnliProcessor,
    "qqp": classifier_data_lib_mem.QqpProcessor,
    "rte": classifier_data_lib_mem.RteProcessor,
    "mrpc": classifier_data_lib_mem.MrpcProcessor,
    "wnli": classifier_data_lib_mem.WnliProcessor,
    "xnli": classifier_data_lib_mem.XnliProcessor,
    }
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    eval_examples = processor.get_dev_examples(FLAGS.input_data_dir)

    labels = [el.label for el in eval_examples]

    predictions = []
    with open(os.path.join(FLAGS.output_dir, 'eval_results.tsv')) as f:
        reader = csv.reader(f, delimiter='\t')
        for el in reader:
            predictions.append([float(x) for x in el])

    print('accuracy_score:', accuracy_score(labels, predictions))
    print('f1_score:', f1_score(labels, predictions))
    print('matthews_corrcoef:', matthews_corrcoef(labels, predictions))

if __name__ == "__main__":
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("output_dir")
  flags.mark_flag_as_required("input_data_dir")
  app.run(main)
