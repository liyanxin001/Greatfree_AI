package ai.greatfree.distributed.rnn.child.app;

import java.util.concurrent.atomic.AtomicInteger;

import org.greatfree.util.IPAddress;
import org.greatfree.util.UtilConfig;

import ai.greatfree.distributed.rnn.RNNConfig;

/**
 * Created: 08/29/2024, Bing Li
 */
public final class ChildProfile
{
	private String trainingDataPath;
	private String trainingDataFile;
	private String trainedDataPath;
	
	private IPAddress coordinatorIP;
	
	private AtomicInteger localAggregationTimes;

	private ChildProfile()
	{
		this.trainingDataPath = RNNConfig.NO_PATH;
		this.trainingDataFile = RNNConfig.NO_FILE;
		this.trainedDataPath = RNNConfig.NO_PATH;
		this.localAggregationTimes = new AtomicInteger(UtilConfig.ZERO);
	}

	private static ChildProfile instance = new ChildProfile();
	
	public static ChildProfile CHILD()
	{
		if (instance == null)
		{
			instance = new ChildProfile();
			return instance;
		}
		else
		{
			return instance;
		}
	}
	
	public void setCoordinatorIP(IPAddress ip)
	{
		this.coordinatorIP = ip;
	}
	
	public IPAddress getCoordinatorIP()
	{
		return this.coordinatorIP;
	}

	public void setTrainingDataPath(String trainingPath)
	{
		this.trainingDataPath = trainingPath;
	}
	
	public String getTrainingDataPath()
	{
		return this.trainingDataPath;
	}
	
	public void setTrainingDataFile(String fileName)
	{
		this.trainingDataFile = this.trainingDataPath + fileName;
	}
	
	public String getTrainingDataFile()
	{
		return this.trainingDataFile;
	}
	
	public void setTrainedDataPath(String trainedPath)
	{
		this.trainedDataPath = trainedPath;
	}
	
	public String getTrainedDataPath()
	{
		return this.trainedDataPath;
	}
	
	public void clearLocalAggregationTimes()
	{
		this.localAggregationTimes.set(UtilConfig.ZERO);
	}
	
	public boolean isLocalAggregationFinished(int maxLocalAggregationTimes)
	{
		if (this.localAggregationTimes.get() >= maxLocalAggregationTimes)
		{
			return true;
		}
		this.localAggregationTimes.incrementAndGet();
		return false;
	}
}
