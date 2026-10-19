package ai.greatfree.distributed.cnn.message;

import edu.greatfree.multicast.message.MulticastResponse;

/**
 * Created: 08/29/2024, Bing Li
 */
public class InitResponse extends MulticastResponse
{
	private static final long serialVersionUID = 4310326660602729256L;

	private String childKey;
	private boolean isDone;

	public InitResponse(String childKey, boolean isDone, String collaboratorKey)
	{
		super(CNNAppID.INIT_RESPONSE, collaboratorKey);
		this.childKey = childKey;
		this.isDone = isDone;
	}
	
	public String getChildKey()
	{
		return this.childKey;
	}

	public boolean isDone()
	{
		return this.isDone;
	}
}
