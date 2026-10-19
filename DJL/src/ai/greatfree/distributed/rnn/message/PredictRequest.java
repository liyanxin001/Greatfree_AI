package ai.greatfree.distributed.rnn.message;

import edu.greatfree.cluster.message.ClusterRequest;
import edu.greatfree.cluster.message.RequestType;

/**
 * Created: 08/29/2024, Bing Li
 */
public class PredictRequest extends ClusterRequest
{
	private static final long serialVersionUID = 7866978416006800710L;
	
	private String prefix;
//	private int numOfPredictions;

//	public PredictRequest(String prefix, int numOfPredictions)
	public PredictRequest(String prefix)
	{
		super(RequestType.BROADCAST_REQUEST, RNNAppID.PREDICT_REQUEST);
		this.prefix = prefix;
//		this.numOfPredictions = numOfPredictions;
	}

	public String getPrefix()
	{
		return this.prefix;
	}

	/*
	public int getNumOfPredictions()
	{
		return this.numOfPredictions;
	}
	*/
}

