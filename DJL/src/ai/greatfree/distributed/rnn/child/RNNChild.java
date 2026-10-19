package ai.greatfree.distributed.rnn.child;

import java.io.IOException;
import java.util.List;
import java.util.logging.Logger;

import org.greatfree.exceptions.RemoteReadException;
import org.greatfree.util.IPAddress;
import org.greatfree.util.TerminateSignal;

import ai.greatfree.distributed.rnn.RNNConfig;
import ai.greatfree.distributed.rnn.child.app.ChildProfile;
import ai.greatfree.distributed.rnn.child.app.RNNTrainer;
import ai.greatfree.distributed.rnn.message.ChildPath;
import ai.greatfree.distributed.rnn.message.ChildPathsRequest;
import ai.greatfree.distributed.rnn.message.ChildPathsResponse;
import ai.greatfree.distributed.rnn.message.GradientsAggregationRequest;
import ai.greatfree.distributed.rnn.message.GradientsAggregationResponse;
import ai.greatfree.distributed.rnn.message.RNNParam;
import ai.greatfree.distributed.rnn.message.TrainingDoneNotification;
import edu.greatfree.cluster.ClusterSpec;
import edu.greatfree.cluster.child.Child;
import edu.greatfree.cluster.child.ChildTask;
import edu.greatfree.exceptions.RootOfflineException;
import edu.greatfree.exceptions.TaskAlreadyExistedException;

/**
 * Created: 08/29/2024, Bing Li
 */
public final class RNNChild
{
	private final static Logger log = Logger.getLogger("ai.greatfree.distributed.rnn.child");

	private Child child;
	
	private RNNChild()
	{
	}
	
	private static RNNChild instance = new RNNChild();
	
	public static RNNChild CLUSTER()
	{
		if (instance == null)
		{
			instance = new RNNChild();
			return instance;
		}
		else
		{
			return instance;
		}
	}

	public void stop() throws ClassNotFoundException, IOException, InterruptedException, RemoteReadException
	{
		RNNTrainer.CHILD().dispose();
		TerminateSignal.SIGNAL().notifyAllTermination();
		this.child.stop();
	}
	
	public void start(ClusterSpec spec, String coordinatorName, ChildTask... tasks) throws ClassNotFoundException, IOException, RemoteReadException, TaskAlreadyExistedException, InterruptedException
	{
		this.child = new Child(spec, tasks);
		try
		{
			this.child.start();
		}
		catch (RootOfflineException e)
		{
			log.info("Root is offline!");
		}
		log.info("Child's address is " + this.child.getAddress());
		
		ChildPath paths = ((ChildPathsResponse)this.child.readRoot(new ChildPathsRequest(RNNConfig.TRAINING_DATA_PATH, RNNConfig.TRAINED_DATA_PATH))).getPath();
		
		ChildProfile.CHILD().setTrainingDataPath(paths.getTrainingDataPath());
		ChildProfile.CHILD().setTrainedDataPath(paths.getTrainedDataPath());
		
		log.info("Training Data Path = " + ChildProfile.CHILD().getTrainingDataPath());
		log.info("Trained Data Path = " + ChildProfile.CHILD().getTrainedDataPath());
		
		IPAddress ip = this.child.getPartnerAddress(coordinatorName);
		log.info("coordinator IP = " + ip);
		if (ip != null)
		{
			ChildProfile.CHILD().setCoordinatorIP(ip);
		}
	}
	
	public String getChildKey()
	{
		return this.child.getChildKey();
	}
	
	public double getGradientsAggregation(List<RNNParam> params) throws ClassNotFoundException, RemoteReadException, IOException
	{
		return ((GradientsAggregationResponse)this.child.read(ChildProfile.CHILD().getCoordinatorIP().getIP(), ChildProfile.CHILD().getCoordinatorIP().getPort(), new GradientsAggregationRequest(this.child.getChildKey(), params))).getNorm();
	}
	
	public void notifyTrainingDone() throws IOException, InterruptedException
	{
		this.child.notify(ChildProfile.CHILD().getCoordinatorIP().getIP(), ChildProfile.CHILD().getCoordinatorIP().getPort(), new TrainingDoneNotification(this.child.getChildKey()));
	}
}

