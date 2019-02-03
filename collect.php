<?php
use VK\Client\VKApiClient;
require "vendor/autoload.php";

function save_arr($filename, $arr) {
    $encodedString = json_encode($arr);
    file_put_contents($filename, $encodedString);
}

function load_arr($filename) {
    $fileContents = file_get_contents($filename);
    $decoded = json_decode($fileContents, true);
    return $decoded;
}

$vk = new VKApiClient();
$token_file = fopen("access_token", "r");
$access_token = fread($token_file, filesize("access_token"));

$collected_groups = array();
$collected_edges_arr = array();


do {
    $start = rand(1, 100000000);
    $end = $start + 10000;
    $possible_group_ids = range($start, $end, 1);
    shuffle($possible_group_ids);
    $group_ids = array_slice($possible_group_ids, 0, 500);
    $groups_response = $vk->groups()->getById($access_token,
        array(
            "group_ids" => $group_ids,
            "fields" => array("members_count")
        ));
    foreach ($groups_response as $item) {
        if (!array_key_exists("deactivated", $item) and $item["type"] == "group" and $item["members_count"] > 100) {
            if (!array_key_exists($item["id"], $collected_groups)) {
                usleep(333333);
                $collected_groups[$item["id"]] = 1;
                $users_response = $vk->groups()->getMembers($access_token, array(
                    "group_id" => $item["id"]
                ));
                $users = $users_response["items"];
                foreach($users as $user) {
                    array_push($collected_edges_arr, array($item["id"], $user));
                }
            }
        }
    }
} while (count($collected_groups) < 30000 or count($collected_edges_arr) < 20000000);


$file = fopen('data.csv', 'w');
fputcsv($file, array('group', 'user', 'target'));
foreach($collected_edges_arr as $edge) {
    fputcsv($file, $edge);
}
fclose($file);
?>