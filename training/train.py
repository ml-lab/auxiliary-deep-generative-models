import numpy as np
from utils import env_paths as paths
from base import Train
import time


class TrainModel(Train):
    def __init__(self, model, anneal_lr=1., anneal_lr_freq=np.inf, output_freq=1, pickle_f_custom_freq=None,
                 f_custom_eval=None):
        super(TrainModel, self).__init__(model, pickle_f_custom_freq, f_custom_eval)
        self.anneal_lr = anneal_lr
        self.output_freq = output_freq
        self.anneal_lr_freq = anneal_lr_freq

    def train_model(self, f_train, train_args, f_test, test_args, f_validate, validation_args,
                    n_train_batches=600, n_valid_batches=1, n_test_batches=1, n_epochs=100):
        self.write_to_logger("### MODEL PARAMS ###")
        self.write_to_logger(self.model.model_info())
        self.write_to_logger("### TRAINING PARAMS ###")
        self.write_to_logger("Train -> %s: %s" % (";".join(train_args['inputs'].keys()), str(train_args['inputs'].values())))
        self.write_to_logger("Test -> %s: %s" % (";".join(test_args['inputs'].keys()), str(test_args['inputs'].values())))
        self.write_to_logger("Anneal LR %0.4f after %i."%(self.anneal_lr, int(self.anneal_lr_freq)))
        self.write_to_logger("### TRAINING MODEL ###")

        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):
            epoch += 1
            if epoch % self.anneal_lr_freq == 0:
                train_args['inputs']['learningrate'] *= self.anneal_lr

            start_time = time.time()
            train_outputs = []
            for i in xrange(n_train_batches):
                train_output = f_train(i, *train_args['inputs'].values())
                train_outputs.append(train_output)
            self.eval_train[epoch] = np.mean(np.array(train_outputs), axis=0)
            self.model.after_epoch()
            end_time = time.time() - start_time

            if epoch % self.output_freq == 0:
                if n_test_batches == 1:
                    self.eval_test[epoch] = f_test(*test_args['inputs'].values())
                else:
                    test_outputs = []
                    for i in xrange(n_test_batches):
                        test_output = f_test(i, *test_args['inputs'].values())
                        test_outputs.append(test_output)
                    self.eval_test[epoch] = np.mean(np.array(test_outputs), axis=0)

                if f_validate is not None:
                    if n_valid_batches == 1:
                        self.eval_validation[epoch] = f_validate(*validation_args['inputs'].values())
                    else:
                        valid_outputs = []
                        for i in xrange(n_valid_batches):
                            valid_output = f_validate(i, *validation_args['inputs'].values())
                            valid_outputs.append(valid_output)
                        self.eval_validation[epoch] = np.mean(np.array(valid_outputs), axis=0)
                else:
                    self.eval_validation[epoch] = [0.] * len(validation_args['outputs'].keys())

                output_str = '[epoch,time,train[%s],test[%s],valid[%s]];%i;%0.2f;' % \
                             (",".join(train_args['outputs'].keys()),
                              ",".join(test_args['outputs'].keys()),
                              ",".join(validation_args['outputs'].keys()), epoch, end_time)
                output_str += ";".join(train_args['outputs'].values()) + ";" + \
                              ";".join(test_args['outputs'].values()) + ";" + \
                              ";".join(validation_args['outputs'].values())
                outputs = [float(o) for o in self.eval_train[epoch]]
                outputs += [float(o) for o in self.eval_test[epoch]]
                outputs += [float(o) for o in self.eval_validation[epoch]]
                output_str %= tuple(outputs)
                self.write_to_logger(output_str)

            if self.pickle_f_custom_freq is not None and epoch % self.pickle_f_custom_freq == 0:
                if self.custom_eval_func is not None:
                    self.custom_eval_func(self.model, paths.get_custom_eval_path(epoch, self.model.root_path))
                self.plot_eval(self.eval_train, train_args['outputs'].keys(), "_train")
                self.plot_eval(self.eval_test, test_args['outputs'].keys(), "_test")
                self.plot_eval(self.eval_validation, validation_args['outputs'].keys(), "_validation")
                self.dump_dicts()
                self.model.dump_model()
        if self.pickle_f_custom_freq is not None:
            self.model.dump_model()
