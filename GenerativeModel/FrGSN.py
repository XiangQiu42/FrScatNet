from GSN import GSN
import os
import copy
import utilss.tools as tool
import matplotlib.pyplot as plt


class FrGSN():
    def __init__(self, parameters, alpha_1, alpha_2):
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

        # specify the different parameters to construct different GSN model
        self.para = {}
        self.Model = {}

        for i in range(len(self.alpha_1)):
            a_1 = self.alpha_1[i]
            a_2 = self.alpha_2[i]

            key = "{0}_{1}".format(a_1, a_2)

            self.para[key] = copy.deepcopy(parameters)
            self.para[key]['exper_path'] = os.path.join(parameters['exper_path'], 'FrGSNs', key)

            self.para[key]['train-norm_filename'] = parameters['train-norm_filename'] + '/' + key
            self.para[key]['test-norm_filename'] = parameters['test-norm_filename'] + '/' + key

            self.Model[key] = GSN(parameters=self.para[key])

    def train(self, epoch_train=80):

        for i in range(len(self.alpha_1)):
            a_1 = self.alpha_1[i]
            a_2 = self.alpha_2[i]
            key = "{0}_{1}".format(a_1, a_2)

            print("-" * 20 + "Start train when alpha_1 = {0} & alpha_2 = {1}".format(a_1, a_2) + "-" * 20)

            self.Model[key].train(epoch_train=epoch_train)

        print("*" * 40 + "Congratulations, you have finished train each pair of frational model!!!" + "-" * 40)

    def compare_result(self, scat_dir):
        """
        Compare different fractional parameter by display their SSIM and PANR score

        Parameters
        ----------
        scat_dir: string
        the dir where the result of scatNet stored

        Returns
        -------

        """
        Scores = {}
        Max_epoch = 0
        for i in range(len(self.alpha_1)):
            a_1 = self.alpha_1[i]
            a_2 = self.alpha_2[i]
            key = "{0}_{1}".format(a_1, a_2)
            Scores[key] = tool.read_run_data(self.Model[key].dir_experiment + '/')
            Max_epoch = max(Max_epoch, Scores[key].shape[0])

        Scores['SCAT'] = tool.read_run_data(scat_dir)

        Max_epoch = 80
        X = range(0, Max_epoch)

        losses = {}
        train_psnrs = {}
        train_ssims = {}
        test_psnrs = {}
        test_ssims = {}

        for key in Scores.keys():
            losses[key] = copy.deepcopy(Scores[key]['loss'])
            train_psnrs[key] = copy.deepcopy(Scores[key]['train_psnr'])
            test_psnrs[key] = copy.deepcopy(Scores[key]['test_psnr'])
            train_ssims[key] = copy.deepcopy(Scores[key]['train_ssim'])
            test_ssims[key] = copy.deepcopy(Scores[key]['test_ssim'])

        plt.figure(1)
        plt.title('Compare Loss')
        for key in losses.keys():
            if key == 'SCAT':
                plt.plot(X, losses[key][:Max_epoch], 'r--', label=key)
                continue
            plt.plot(X, losses[key][:Max_epoch], label=key)
        plt.legend()
        plt.ylabel('LOSS')
        plt.xlabel('epoch')
        plt.savefig(os.path.join("/home/qiuxiang/experiments", 'Resluts_loss.jpg'))
        plt.show()

        plt.figure(2)
        # 对比 train_psnrs
        plt.subplot(221)
        for key in train_psnrs.keys():
            if key == 'SCAT':
                plt.plot(X, train_psnrs[key][:Max_epoch], 'r--', label=key)
                continue
            plt.plot(X, train_psnrs[key][:Max_epoch], label=key)
        plt.ylabel('train_PSNR')
        plt.xlabel('epoch')
        plt.legend()

        # 对比 train_ssims
        plt.subplot(222)
        for key in train_ssims.keys():
            if key == 'SCAT':
                plt.plot(X, train_ssims[key][:Max_epoch], 'r--', label=key)
                continue
            plt.plot(X, train_ssims[key][:Max_epoch], label=key)
        plt.legend()
        plt.ylabel('train_SSIM')
        plt.xlabel('epoch')

        # 对比 test_psnrs
        plt.subplot(223)
        for key in test_psnrs.keys():
            if key == 'SCAT':
                plt.plot(X, test_psnrs[key][:Max_epoch], 'r--', label=key)
                continue
            plt.plot(X, test_psnrs[key][:Max_epoch], label=key)
        plt.legend()
        plt.ylabel('test_PSNR')
        plt.xlabel('epoch')

        # 对比 test_psnrs
        plt.subplot(224)
        for key in test_ssims.keys():
            if key == 'SCAT':
                plt.plot(X, test_ssims[key][:Max_epoch], 'r--', label=key)
                continue
            plt.plot(X, test_ssims[key][:Max_epoch], label=key)
        plt.legend()
        plt.ylabel('test_SSIM')
        plt.xlabel('epoch')

        plt.savefig(os.path.join("/home/qiuxiang/experiments", 'Resluts_ssim&psnr.jpg'))
        plt.show()
