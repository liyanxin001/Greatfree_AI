package ai.greatfree.distributed.cnn.root;

import java.util.concurrent.atomic.AtomicInteger;

import org.greatfree.util.UtilConfig;

import ai.greatfree.distributed.rnn.RNNConfig;
import ai.greatfree.distributed.rnn.message.ChildPath;

/**
 * Created: 08/29/2024, Bing Li
 */
final class Paths
{
	private AtomicInteger currentCount;
	
	private Paths()
	{
	}

	private static Paths instance = new Paths();
	
	public static Paths ROOT()
	{
		if (instance == null)
		{
			instance = new Paths();
			return instance;
		}
		else
		{
			return instance;
		}
	}

	public void init()
	{
		this.currentCount = new AtomicInteger(0);
	}

	public synchronized ChildPath getPath(String trainingDataPath, String trainedDataPath)
	{
		String childID = RNNConfig.CHILD + this.currentCount.getAndIncrement();
		trainingDataPath = trainingDataPath + childID + UtilConfig.FORWARD_SLASH;
		trainedDataPath = trainedDataPath + childID + UtilConfig.FORWARD_SLASH;
		return new ChildPath(trainingDataPath, trainedDataPath);
	}
}
