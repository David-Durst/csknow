let helloWorld: string = "Hellow world2"
console.log(helloWorld)

// Import required AWS SDK clients and commands for Node.js
const { S3Client, ListObjectsCommand } = require("@aws-sdk/client-s3");
const {CognitoIdentityClient} = require("@aws-sdk/client-cognito-identity");
const {fromCognitoIdentityPool} = require("@aws-sdk/credential-provider-cognito-identity");

// Set the AWS region
const REGION = "us-east-1"; //e.g. "us-east-1"

// Create the parameters for the bucket
const bucketParams = { Bucket: "csknow" };

const cognitoIdentityClient = new CognitoIdentityClient({
    region: REGION
});

// Create S3 service object
const s3 = new S3Client({
    region: REGION,
    credentials: fromCognitoIdentityPool({
        client: cognitoIdentityClient,
        identityPoolId: "us-east-1:b97cc6dd-33b3-4672-a48b-9fa1876d8c78"
    })
});

const run = async () => {
    try {
        const data = await s3.send(new ListObjectsCommand(bucketParams));
        console.log("Success", data);
    } catch (err) {
        console.log("Error", err);
    }
};
run();