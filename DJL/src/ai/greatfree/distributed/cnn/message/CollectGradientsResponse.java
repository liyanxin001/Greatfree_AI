package ai.greatfree.distributed.cnn.message;

import java.util.List;

import edu.greatfree.multicast.message.MulticastResponse;

/**
 * Created: 09/01/2024, Bing Li
 */
public class CollectGradientsResponse extends MulticastResponse
{
	private static final long serialVersionUID = 6746558468470547224L;
	
	private List<CNNParam> params;

	public CollectGradientsResponse(List<CNNParam> params, String collaboratorKey)
	{
		super(CNNAppID.COLLECT_GRADIENTS_RESPONSE, collaboratorKey);
		this.params = params;
	}

	public List<CNNParam> getParams()
	{
		return this.params;
	}
}
