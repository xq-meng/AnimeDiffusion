import logging


class Palette:
    
    def __init__(self, options):
        # member variables
        self.options = options
        self.epoch = 0
        logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

    def train_step(self):
        pass

    def train(self):
        while self.epoch < self.options['train']['n_epoch']:
            self.epoch += 1
            self.train_step()
            logging.info("Epoch = %d", self.epoch - 1)

        logging.info('End of traing. Epoch = %d', self.options['train']['n_epoch'])

        
if __name__ == '__main__':
    option = {'train': {'n_epoch': 20}}
    p = Palette(options=option)
    p.train()