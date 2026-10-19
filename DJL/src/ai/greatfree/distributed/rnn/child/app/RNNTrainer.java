package ai.greatfree.distributed.rnn.child.app;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.logging.Logger;

import org.greatfree.exceptions.RemoteReadException;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Pair;
import ai.greatfree.distributed.rnn.RNNConfig;
import ai.greatfree.distributed.rnn.child.RNNChild;
import ai.greatfree.distributed.rnn.utils.RNNTools;
import ai.greatfree.distributed.rnn.utils.SequentialTextLoader;
import ai.greatfree.utils.RNNModelScratch;
import ai.greatfree.utils.Vocab;
import ai.greatfree.utils.Functions.TriFunction;

/**
 * Created: 08/29/2024, Bing Li
 */
public final class RNNTrainer
{
	private NDManager manager;
	private SequentialTextLoader loader;
	private Pair<ArrayList<NDList>, Vocab> timeMachine;
	private List<NDList> trainIter;
	private Vocab vacabularies;
	private TriFunction<Integer, Integer, Device, NDList> getParamsFn;
	private TriFunction<Integer, Integer, Device, NDList> initRNNStateFn;
	private TriFunction<NDArray, NDList, NDList, Pair<NDArray, NDList>> rnnFn;
	private Device device;
	private int numEpochs;
	private RNNModelScratch net;
	private Function<String, String> predictor;
	
	private RNNTrainer()
	{
	}

	private static RNNTrainer instance = new RNNTrainer();
	
	public static RNNTrainer CHILD()
	{
		if (instance == null)
		{
			instance = new RNNTrainer();
			return instance;
		}
		else
		{
			return instance;
		}
	}
	
	public void dispose()
	{
		this.manager.close();
	}

	private final static Logger log = Logger.getLogger("ai.greatfree.distributed.rnn.child.app");

	public void init() throws IOException, Exception
	{
		this.manager = NDManager.newBaseManager();
		
		this.loader = new SequentialTextLoader.SequentialTextLoaderBuilder()
//				.fileName(ChildProfile.CHILD().getTrainingDataPath())
				.fileName(ChildProfile.CHILD().getTrainingDataFile())
				.startIndex(RNNConfig.LOAD_START_INDEX)
				.endIndex(RNNConfig.LOAD_END_INDEX)
				.batchSize(RNNConfig.TRAINING_BATCH_SIZE)
				.numSteps(RNNConfig.TRAINING_NUM_STEPS)
				.useRandomIter(RNNConfig.USE_RANDOM_ITERATIVE)
				.maxTokens(RNNConfig.MAX_TOKENS)
				.manager(this.manager)
				.build();

		if (this.loader == null)
		{
			log.info("1) loader is NULL!!!");
		}
		
//		this.timeMachine = RNNTools.loadTrainingData(RNNConfig.TRAINING_BATCH_SIZE, RNNConfig.TRAINING_NUM_STEPS, RNNConfig.USE_RANDOM_ITERATIVE, RNNConfig.MAX_TOKENS, this.manager);
		this.timeMachine = RNNTools.loadTrainingData(this.loader);
		this.trainIter = this.timeMachine.getKey();
		this.vacabularies = this.timeMachine.getValue();
		
		this.getParamsFn = (a, b, c) -> RNNTools.getParams(a, b, c, this.manager);
		this.initRNNStateFn = (a, b, c) -> RNNTools.initRNNState(a, b, c, this.manager);
		this.rnnFn = (a, b, c) -> RNNTools.rnn(a, b, c);
		this.device = this.manager.getDevice();
		
		this.numEpochs = Integer.getInteger(RNNConfig.MAX_EPOCH, RNNConfig.NUM_EPOCHS);
		this.net = new RNNModelScratch(this.vacabularies.length(), RNNConfig.NUM_HIDDENS, this.device, this.getParamsFn, this.initRNNStateFn, this.rnnFn);
	}
	
	public RNNModelScratch getRNN()
	{
		return this.net;
	}

	public void train() throws ClassNotFoundException, RemoteReadException, IOException, InterruptedException
	{
		this.predictor = RNNTools.train(this.net, this.trainIter, this.vacabularies, RNNConfig.LEARNING_RATE, this.numEpochs, this.device, RNNConfig.USE_RANDOM_ITERATIVE, this.manager);
		RNNChild.CLUSTER().notifyTrainingDone();
	}
	
	public String predict(String prefix)
	{
		return this.predictor.apply(prefix);
	}
}
