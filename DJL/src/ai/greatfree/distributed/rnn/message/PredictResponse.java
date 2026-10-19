package ai.greatfree.distributed.rnn.message;

import edu.greatfree.multicast.message.MulticastResponse;

/**
 * Created: 08/29/2024, Bing Li
 */
public class PredictResponse extends MulticastResponse
{
	private static final long serialVersionUID = -5979537489199424534L;
	
	private String prediction;

	public PredictResponse(String prediction, String collaboratorKey)
	{
		super(RNNAppID.PREDICT_RESPONSE, collaboratorKey);
		this.prediction = prediction;
	}

	public String getPrediction()
	{
		return this.prediction;
	}
}

