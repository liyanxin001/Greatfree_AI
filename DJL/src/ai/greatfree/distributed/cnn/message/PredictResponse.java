package ai.greatfree.distributed.cnn.message;

import edu.greatfree.multicast.message.MulticastResponse;

/**
 * Created: 08/29/2024, Bing Li
 */
public class PredictResponse extends MulticastResponse
{
	private static final long serialVersionUID = -5979537489199424534L;
	
	private String label;
	private float confidence;

	public PredictResponse(String label, float confidence, String collaboratorKey)
	{
		super(CNNAppID.PREDICT_RESPONSE, collaboratorKey);
		this.label = label;
		this.confidence = confidence;
	}

    public String getLabel() 
    {
    	return this.label;
    }
	public float getConfidence() {
		return confidence;
	}

	public void setConfidence(float confidence) {
		this.confidence = confidence;
	}
}

