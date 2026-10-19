package ai.greatfree.distributed.rnn;

import org.greatfree.util.UtilConfig;

/**
 * Created: 08/28/2024, Bing Li
 */
public final class RNNConfig
{
//	public final static String HOME_PATH = "/home/greatfree/DJL/Data/";
//	public final static String HOME_PATH = "/home/libing/Wind/AI/Data/";
//	public final static String HOME_PATH = "/home/greatfree/Wind/AI/Data/";
	public final static String HOME_PATH = "/home/liyanxin/AI/Data/";
	public final static String SOURCE_DATA_PATH = HOME_PATH + "SourceData/";
	public final static String TRAINING_DATA_PATH = HOME_PATH + "TrainingData/";
	public final static String TRAINED_DATA_PATH = HOME_PATH + "TrainedData/";
	
	public final static String CHILD = "Child";
	public final static String NO_PATH = UtilConfig.EMPTY_STRING;
	public final static String NO_FILE = UtilConfig.EMPTY_STRING;
	
	public final static int TRAINING_BATCH_SIZE = 32;
	public final static int TRAINING_NUM_STEPS = 35;
	public final static boolean USE_RANDOM_ITERATIVE = false;
	public final static int MAX_TOKENS = 10000;
//	public final static int MAX_TOKENS = 1000;
	public final static int NUM_HIDDENS = 512;
	public final static int NUM_OF_EPOCHS = 100;
	public final static int LEARNING_RATE = 1;

	public final static String MAX_EPOCH = "MAX_EPOCH";
	public final static int NUM_EPOCHS = 500;
	public final static int NUM_PERIODS = 50;
	
	public final static String REGEXPRESS_FILTER = "[^A-Za-z]+";
	
	public final static String TOKENIZE_BY_WORD = "word";
	public final static String TOKENIZE_BY_CHAR = "char";
	
	public final static String ERROR_TOKEN = "ERROR: unknown token type: ";
	
	public final static int LOAD_START_INDEX = 0;
//	public final static int LOAD_END_INDEX = 10000;
	public final static int LOAD_END_INDEX = 3220;
//	public final static int LOAD_END_INDEX = 4999;
//	public final static int LOAD_END_INDEX = 2000;
//	public final static int LOAD_END_INDEX = 9;
	
	public final static int ACCUMULATOR_NUM = 2;
	public final static int THETA = 1;
	public final static int SQUARE = 2;
	public final static int LOCAL_AGGREGATION_TIME = 5;
	public final static int UPDATE_BATCH_SIZE = 1;
	
	public final static double NO_NORM = -1d;
	
	public final static String TRAINING_COORDINATOR = "TrainingCoordinator";
	public final static int TRAINING_COORDINATOR_PORT = 9000;
	public final static String TRAINING_ROOT = "TraingingRoot";
	public final static int TRAINING_ROOT_PORT = 9050;
	public final static int TRAINING_CHILD_PORT = 9080;
	
//	public final static String REGISTRY_IP = "192.168.2.56";
	public final static String REGISTRY_IP = "192.168.1.25";
	public final static int REGISTRY_PORT = 8941;
}

