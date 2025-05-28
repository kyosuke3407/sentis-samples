using UnityEngine;
using Unity.Sentis;
using System;
using System.Globalization;
using Unity.Mathematics;
using UnityEngine.UI;

public class AttentionMeshRunner : MonoBehaviour
{
    [SerializeField]private ModelAsset AttentionAsset;
    private Model AttentionMeshModel;
    private Worker worker;
    private Tensor<float> inputTensor, outputTensor;
    private TextureTransform transform;

    [SerializeField] private Texture2D tex;
    public RawImage rawImage;
    private RenderTexture renderTexture;


    void Start()
    {
        AttentionMeshModel = ModelLoader.Load(AttentionAsset);
        var graph = new FunctionalGraph();

        var input = graph.AddInput(AttentionMeshModel, 0);
        var outputs = Functional.Forward(AttentionMeshModel,　input);


        AttentionMeshModel = graph.Compile(outputs);
        ////var AttentionMeshModel = graph.Compile(resized);
        //var AttentionMeshModel = graph.Compile(boxN,selectedAnchor);


        worker = new Worker(AttentionMeshModel, BackendType.GPUCompute);
        inputTensor = new Tensor<float>(new TensorShape(1, 3, 192, 192));
        transform = new TextureTransform()
            .SetDimensions(192, 192, 3)
            .SetTensorLayout(TensorLayout.NCHW);


        TextureConverter.ToTensor(tex, inputTensor, transform);
        worker.SetInput(0, inputTensor);

        worker.Schedule();

        var output = AttentionMeshModel.outputs;
        foreach (var outputTensor in output)
        {
            Debug.Log($"Output name: {outputTensor.name}, shape: {outputTensor.index}");
        }

        var output0 = worker.PeekOutput(0) as Tensor<float>;
        var output1 = worker.PeekOutput(1) as Tensor<float>;
        var output2 = worker.PeekOutput(2) as Tensor<float>;
        var output3 = worker.PeekOutput(3) as Tensor<float>;
        var output4 = worker.PeekOutput(4) as Tensor<float>;
        var output5 = worker.PeekOutput(5) as Tensor<float>;
        var output6 = worker.PeekOutput(6) as Tensor<float>;

        Debug.Log($"Output0 shape: {output0.shape}");
        Debug.Log($"Output1 shape: {output1.shape}");
        Debug.Log($"Output2 shape: {output2.shape}");
        Debug.Log($"Output3 shape: {output3.shape}");
        Debug.Log($"Output4 shape: {output4.shape}");
        Debug.Log($"Output5 shape: {output5.shape}");
        Debug.Log($"Output6 shape: {output6.shape}");

        //foreach (var output in AttentionMeshModel.Output)
        //{
        //    Debug.Log($"Output name: {output.name}, shape: {output.shape}");
        //}
        //var output = worker.PeekOutput(0) as Tensor<float>;
        //if (output != null)
        //{
        //    float[] values = output.DownloadToArray();
        //    // 先頭10個だけ表示（大量の場合の例）
        //    for (int i = 0; i < Mathf.Min(10, values.Length); i++)
        //    {
        //        Debug.Log($"output[{i}] = {values[i]}");
        //    }
        //    Debug.Log($"output length: {values.Length}");
        //}
        //else
        //{
        //    Debug.LogWarning("output is null or not a Tensor<float>");
        //}

        //var output = worker.PeekOutput(0) as Tensor<float>;
        //Debug.Log("Output shape: " + output.shape);

        //renderTexture = TextureConverter.ToTexture(output);
        //rawImage.texture = renderTexture;


    }


    void Update()
    {
        //TextureConverter.ToTensor(tex, inputTensor,transform);
        //worker.SetInput(0, inputTensor);

        //worker.Schedule();

        //var indexFace = worker.PeekOutput(0) as Tensor<int>;
        //var score = worker.PeekOutput(1) as Tensor<float>;
        //var Box = worker.PeekOutput(2) as Tensor<float>;

        //var cpuIndex = indexFace.ReadbackAndClone();
        //var cpuScore = score.ReadbackAndClone();
        //var cpuBox = Box.ReadbackAndClone();

        //Debug.Log(cpuScore);
        //Debug.Log(cpuBox);
        //Debug.Log(cpuIndex);
        

    }
}
