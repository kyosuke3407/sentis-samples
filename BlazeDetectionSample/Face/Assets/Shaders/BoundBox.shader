Shader "Unlit/BoundBox"
{
    Properties
    {
        _MainTex("Texture", 2D) = "white"{}
        _Color("Box Color", Color) = (1, 0, 0, 1)
        _Box("Bounding Box (x, y, w, h)", Vector) = (0.2, 0.2, 0.4, 0.3)
        _LineWidth("Line Width", Float) = 0.005
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_ST;
            fixed4 _Color;
            float4 _Box;
            float _LineWidth;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float2 uv = i.uv;
                float2 boxMin = _Box.xy / 128;
                float2 boxMax = (_Box.xy + _Box.zw) / 128;

                float lw = _LineWidth;

                bool inBox =
                    (abs(uv.x - boxMin.x) < lw && uv.y >= boxMin.y && uv.y <= boxMax.y) || // left
                    (abs(uv.x - boxMax.x) < lw && uv.y >= boxMin.y && uv.y <= boxMax.y) || // right
                    (abs(uv.y - boxMin.y) < lw && uv.x >= boxMin.x && uv.x <= boxMax.x) || // bottom
                    (abs(uv.y - boxMax.y) < lw && uv.x >= boxMin.x && uv.x <= boxMax.x);   // top

                fixed4 texColor = tex2D(_MainTex, uv);
                return inBox ? _Color : texColor;
            }
            ENDCG
        }
    }
}
