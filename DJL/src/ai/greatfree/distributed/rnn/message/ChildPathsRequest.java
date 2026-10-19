package ai.greatfree.distributed.rnn.message;

import edu.greatfree.cluster.message.ChildRequest;

/**
 * Created: 08/29/2024, Bing Li
 */
public class ChildPathsRequest extends ChildRequest
{
	private static final long serialVersionUID = 842147357933248500L;
	
	private String trainingDataPath;
	private String trainedDataPath;

	public ChildPathsRequest(String trainingDataPath, String trainedDataPath)
	{
		super(RNNAppID.CHILD_PATHS_REQUEST);
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

