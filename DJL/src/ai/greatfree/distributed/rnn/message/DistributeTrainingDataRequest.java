package ai.greatfree.distributed.rnn.message;

import java.util.List;

import edu.greatfree.cluster.message.ClusterRequest;

/**
 * Created: 08/28/2024, Bing Li
 */
public class DistributeTrainingDataRequest extends ClusterRequest
{
	private static final long serialVersionUID = 6632905411177612028L;

	private String childKey;
	private String fileName;
	private List<String> texts;

	public DistributeTrainingDataRequest(String childKey, String fileName, List<String> texts)
	{
		super(RNNAppID.DISTRIBUTE_TRAINING_DATA_REQUEST, childKey);
		this.childKey = childKey;
		this.fileName = fileName;
		this.texts = texts;
	}
	
	public String getChildKey()
	{
		return this.childKey;
	}
	
	public String getFileName()
	{
		return this.fileName;
	}
	
	public List<String> getTexts()
	{
		return this.texts;
	}
}

