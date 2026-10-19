package ai.greatfree.distributed.rnn.message;

import java.io.Serializable;

/**
 * Created: 08/29/2024, Bing Li
 */
public class ChildPath implements Serializable
{
	private static final long serialVersionUID = 7447594534862130293L;
	
	private String trainingDataPath;
	private String trainedDataPath;
	
	public ChildPath(String trainingDataPath, String trainedDataPath)
	{
		this.trainingDataPath = trainingDataPath;
		this.trainedDataPath = trainedDataPath;
	}
	
	public String getTrainingDataPath()
	{
		return this.trainingDataPath;
	}

	public String getTrainedDataPath()
	{
		return this.trainedDataPath;
	}
}
