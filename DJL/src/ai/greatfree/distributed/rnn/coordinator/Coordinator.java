package ai.greatfree.distributed.rnn.coordinator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import org.greatfree.exceptions.NullClassConversionException;
import org.greatfree.exceptions.RemoteReadException;
import org.greatfree.framework.container.cs.multinode.message.ShutdownServerNotification;
import org.greatfree.util.FileManager;
import org.greatfree.util.IPAddress;
import org.greatfree.util.TerminateSignal;
import org.greatfree.util.Tools;
import org.greatfree.util.UtilConfig;

import ai.greatfree.distributed.rnn.RNNConfig;
import ai.greatfree.distributed.rnn.message.CollectGradientsRequest;
import ai.greatfree.distributed.rnn.message.CollectGradientsResponse;
import ai.greatfree.distributed.rnn.message.DistributeTrainingDataRequest;
import ai.greatfree.distributed.rnn.message.DistributeTrainingDataResponse;
import ai.greatfree.distributed.rnn.message.InitRequest;
import ai.greatfree.distributed.rnn.message.InitResponse;
import ai.greatfree.distributed.rnn.message.PredictRequest;
import ai.greatfree.distributed.rnn.message.PredictResponse;
import ai.greatfree.distributed.rnn.message.RNNParam;
import ai.greatfree.distributed.rnn.message.StartTrainingNotification;
import edu.greatfree.cluster.message.ChildrenKeysRequest;
import edu.greatfree.cluster.message.ChildrenKeysResponse;
import edu.greatfree.container.server.PeerContainer;
import edu.greatfree.container.server.PeerSpec;
import edu.greatfree.exceptions.TaskAlreadyExistedException;
import edu.greatfree.framework.cluster.multicast.message.ShutdownChildrenNotification;
import edu.greatfree.framework.cluster.multicast.message.ShutdownRootNotification;

/**
 * Created: 09/01/2024, Bing Li
 */
final class Coordinator
{
	private PeerContainer peer;
	private IPAddress rootAddress;
	private Set<String> childrenKeys;
	private AtomicInteger trainingDoneCount;
	
	public Coordinator()
	{
	}
	
	private static Coordinator instance = new Coordinator();
	
	public static Coordinator TRAIN()
	{
		if (instance == null)
		{
			instance = new Coordinator();
			return instance;
		}
		else
		{
			return instance;
		}
	}

	public void stop() throws ClassNotFoundException, IOException, InterruptedException, RemoteReadException
	{
		TerminateSignal.SIGNAL().notifyAllTermination();
		this.peer.stop();
	}

	public void start(PeerSpec spec, String rootName) throws IOException, TaskAlreadyExistedException, ClassNotFoundException, RemoteReadException
	{
		this.peer = new PeerContainer(spec);
		this.peer.start();
		this.rootAddress = this.peer.getPartnerAddressByName(rootName);
//		this.childrenKeys = new HashSet<String>(((ChildrenKeysResponse)this.peer.read(this.rootAddress.getIP(), this.rootAddress.getPort(), new ChildrenKeysRequest())).getChildrenKeys());
		this.trainingDoneCount = new AtomicInteger(UtilConfig.ZERO);
	}
	
	public List<RNNParam> getAllParamsExcept(String childKey) throws ClassNotFoundException, RemoteReadException, IOException, NullClassConversionException
	{
		Set<String> childrens = new HashSet<String>(this.childrenKeys);
		childrens.remove(childKey);
		List<CollectGradientsResponse> cgrs = this.peer.read(this.rootAddress.getIP(), this.rootAddress.getPort(), new CollectGradientsRequest(childrens), CollectGradientsResponse.class);
		List<RNNParam> params = new ArrayList<RNNParam>();
		for (CollectGradientsResponse entry : cgrs)
		{
			params.addAll(entry.getParams());
		}
		return params;
	}
	
	public boolean isDone(String childKey)
	{
		if (this.childrenKeys.contains(childKey))
		{
			this.trainingDoneCount.incrementAndGet();
			if (this.trainingDoneCount.get() >= this.childrenKeys.size())
			{
				return true;
			}
		}
		return false;
	}
	
	public int getDoneCount()
	{
		return this.trainingDoneCount.get();
	}

	public void distributeTrainingData() throws IOException, ClassNotFoundException, RemoteReadException, NullClassConversionException
	{
		String textName;
		String fileName;
		List<String> texts;
		List<DistributeTrainingDataResponse> dtdrs;
		this.childrenKeys = new HashSet<String>(((ChildrenKeysResponse)this.peer.read(this.rootAddress.getIP(), this.rootAddress.getPort(), new ChildrenKeysRequest())).getChildrenKeys());
		for (String entry : this.childrenKeys)
		{
			System.out.println("Assign text data for the child, " + entry);
			textName = Tools.INPUT.nextLine();
			fileName = RNNConfig.SOURCE_DATA_PATH + textName;
//			texts = FileManager.readText(fileName);
			texts = FileManager.readLinesByIndex(fileName, RNNConfig.LOAD_START_INDEX, RNNConfig.LOAD_END_INDEX);
			dtdrs = this.peer.read(this.rootAddress.getIP(), this.rootAddress.getPort(), new DistributeTrainingDataRequest(entry, textName, texts), DistributeTrainingDataResponse.class);
			for (DistributeTrainingDataResponse res : dtdrs)
			{
				if (!res.isSucceeded())
				{
					System.out.println("Data is NOT distributed to the child, " + entry + ", successfully!");
				}
				else
				{
					System.out.println("Data is distributed to the child, " + entry + ", successfully!");
				}
			}
		}
	}
	
	public void initializeTraining() throws ClassNotFoundException, RemoteReadException, IOException, NullClassConversionException
	{
		List<InitResponse> irs = this.peer.read(this.rootAddress.getIP(), this.rootAddress.getPort(), new InitRequest(), InitResponse.class);
		for (InitResponse entry : irs)
		{
			if (!entry.isDone())
			{
				System.out.println("Child, " + entry.getChildKey() + ", is NOT initialized successfully!");
			}
			else
			{
				System.out.println("Child, " + entry.getChildKey() + ", is initialized successfully!");
			}
		}
	}
	
	public void startTraining() throws IOException, InterruptedException
	{
		this.peer.syncNotify(this.rootAddress.getIP(), this.rootAddress.getPort(), new StartTrainingNotification());
	}
	
	public void predict() throws ClassNotFoundException, RemoteReadException, IOException, NullClassConversionException
	{
		System.out.println("Tell me your prefix: ");
		String prefix = Tools.INPUT.nextLine();
		List<PredictResponse> prs = this.peer.read(this.rootAddress.getIP(), this.rootAddress.getPort(), new PredictRequest(prefix), PredictResponse.class);
		int index = 0;
		for (PredictResponse entry : prs)
		{
			System.out.println(index++ + ") " + entry.getPrediction());
		}
	}
	
	public void shutdownChildren() throws IOException, InterruptedException
	{
		this.peer.syncNotify(this.rootAddress.getIP(), this.rootAddress.getPort(), new ShutdownChildrenNotification());
	}
	
	public void shutdownRoot() throws IOException, InterruptedException
	{
		this.peer.syncNotify(this.rootAddress.getIP(), this.rootAddress.getPort(), new ShutdownRootNotification());
	}
	
	public void shutdownRegistryServer() throws IOException, InterruptedException
	{
		this.peer.syncNotify(RNNConfig.REGISTRY_IP, RNNConfig.REGISTRY_PORT, new ShutdownServerNotification());
	}
}

