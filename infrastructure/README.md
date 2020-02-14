
### Data Pipeline

Create a pipeline node.  Note, the user provides the {pipeline-name} & {pipeline-code}
```bash
>> aws datapipeline create-pipeline --name {pipeline-name} --unique-id {pipeline-code}
<< {pipeline-id}
```

<br>

Submit the pipeline design
```bash
>> aws datapipeline put-pipeline-definition --pipeline-id {pipeline-id} --pipeline-definition file://....json
```

<br>

Add tags to the pipeline, if required.
```bash
>> aws datapipeline add-tags --pipeline-id {pipeline-id} --tags key=...,value=...
```

<br>

Activate the pipeline.
```bash
>> aws datapipeline activate-pipeline --pipeline-id {pipeline-id}
```

<br>
<br>

### EC2

* https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-launch-templates.html
* https://docs.aws.amazon.com/cli/latest/reference/ec2/create-launch-template.html
* https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/user-data.html

<br>

The EC2 template creation command

```bash
>> aws ec2 create-launch-template --launch-template-name TemplateForDocker --version-description 0.1 --launch-template-data file://template-data.json
```
leads to an output of the form
```bash
<<
{
    "LaunchTemplate": {
        "LatestVersionNumber": 1,
        "LaunchTemplateId": "{template-id}",
        "LaunchTemplateName": "...",
        "DefaultVersionNumber": 1,
        "CreatedBy": "arn:aws:sts::...",
        "CreateTime": "yyyy-MM-ddT2HH:mm:ss.SSSZ"
    }
}
```

Hence, to launch an instance from a template, with user data, ...

```bash
>> aws ec2 run-instances --launch-template LaunchTemplateId={template-id} --user-data file://user-data.txt
```

<br>

To extract an instance's data for template launching purposes
```bash
>> aws ec2 get-launch-template-data --instance-id {instance-id} --query "LaunchTemplateData"
```

<br>

To terminate an instance
```bash
>> aws ec2 terminate-instances --instance-ids {instance-id}
```


### Code Build

* https://docs.aws.amazon.com/codebuild/latest/userguide/create-project.html#create-project-cli
* https://docs.aws.amazon.com/cli/latest/reference/codebuild/create-project.html
* https://docs.aws.amazon.com/cli/latest/reference/codebuild/index.html#cli-aws-codebuild

Generate a skeleton project template via
```bash
aws codebuild create-project --generate-cli-skeleton
```

The project file [create-project.json](/infrastructure/codebuild/create-project.json) is under development.
