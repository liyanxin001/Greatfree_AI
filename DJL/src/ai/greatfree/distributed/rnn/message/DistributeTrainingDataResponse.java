package ai.greatfree.distributed.rnn.message;

import edu.greatfree.multicast.message.MulticastResponse;

/**
 * Created: 08/29/2024, Bing Li
 */
public class DistributeTrainingDataResponse extends MulticastResponse
{
	private static final long serialVersionUID = -2504521024760868068L;
	
	private boolean isSucceeded;

	public DistributeTrainingDataResponse(boolean isSucceeded, String childKey, String collaboratorKey)
	{
		super(RNNAppID.DISTRIBUTE_TRAINING_DATA_RESPONSE, childKey, collaboratorKey);
		this.isSucceeded = isSucceeded;
	}

	public boolean isSucceeded()
	{
		return this.isSucceeded;
	}
}
