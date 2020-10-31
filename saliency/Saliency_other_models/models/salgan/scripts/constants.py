# Work space directory
HOME_DIR = '/media/dberga/DATA/repos/BIOvsDL/modelos/salgan/'

# Path to SALICON raw data
pathToImages = '/media/dberga/DATA/repos/BIOvsDL/modelos/salgan/images'
pathToMaps = '/media/dberga/DATA/repos/BIOvsDL/modelos/salgan/saliency'
pathToFixationMaps = '/media/dberga/DATA/repos/BIOvsDL/modelos/salgan/fixation'

# Path to processed data
pathOutputImages = '/media/dberga/DATA/repos/BIOvsDL/modelos/salgan/salicon_data/images320x240'
pathOutputMaps = '/media/dberga/DATA/repos/BIOvsDL/modelos/salgan/salicon_data/saliency320x240'
pathToPickle = '/media/dberga/DATA/repos/BIOvsDL/modelos/salgan/salicon_data/320x240'

# Path to pickles which contains processed data
TRAIN_DATA_DIR = '/media/dberga/DATA/repos/BIOvsDL/modelos/salgan/salicon_data/320x240/fix_trainData.pickle'
VAL_DATA_DIR = '/media/dberga/DATA/repos/BIOvsDL/modelos/salgan/salicon_data/320x240/fix_validationData.pickle'
TEST_DATA_DIR = '/media/dberga/DATA/repos/BIOvsDL/modelos/salgan/salicon_data/256x192/testData.pickle'

# Path to vgg16 pre-trained weights
PATH_TO_VGG16_WEIGHTS = '/media/dberga/DATA/repos/BIOvsDL/modelos/salgan/vgg16.pkl'

# Input image and saliency map size
INPUT_SIZE = (256, 192)

# Directory to keep snapshots
DIR_TO_SAVE = 'test'
